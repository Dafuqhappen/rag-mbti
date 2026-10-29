"""Замер: ускоряет ли параллельный клиент наш RAG на локальной ollama.

Прогоняет вопросы golden set через полный answer() (поиск + генерация qwen)
с разной степенью одновременности и сравнивает:
  - общее время прогона (throughput — сколько вопросов в минуту);
  - задержку одного ответа (latency — сколько ждёт каждый пользователь).

Запуск:  .venv/bin/python src/parallel_bench.py            # 1, 2, 4, 8
         .venv/bin/python src/parallel_bench.py 1 4        # выбранные уровни

answer() синхронный, поэтому в asyncio он уходит через to_thread — как в bot.py.
"""
import asyncio
import statistics
import sys
import time

from evaluate import load_golden
from rag import answer

LEVELS = (1, 2, 4, 8)
REQUEST_TIMEOUT_SEC = 120


async def timed_answer(question: str, gate: asyncio.Semaphore) -> float | None:
    """Задержка одного ответа в секундах; None — таймаут.

    Отсчёт идёт с момента, когда запрос прошёл турникет: так видно, сколько
    отвечает сам сервер, без ожидания своей очереди у нас.
    """
    async with gate:
        start = time.perf_counter()
        try:
            # Таймаут обрывает ОЖИДАНИЕ, но не поток: to_thread нельзя прервать,
            # запрос к ollama доработает в фоне. Для замера это допустимо.
            await asyncio.wait_for(asyncio.to_thread(answer, question), REQUEST_TIMEOUT_SEC)
        except TimeoutError:
            return None
        return time.perf_counter() - start


async def run_level(questions: list[str], concurrency: int) -> dict:
    gate = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    latencies = await asyncio.gather(*(timed_answer(q, gate) for q in questions))
    wall = time.perf_counter() - start
    done = [x for x in latencies if x is not None]
    return {
        "concurrency": concurrency,
        "wall": wall,
        "per_min": len(done) / wall * 60,
        "p50": statistics.median(done),
        "max": max(done),
        "timeouts": len(latencies) - len(done),
    }


def main() -> None:
    levels = tuple(int(a) for a in sys.argv[1:]) or LEVELS
    questions = [case["query"] for case in load_golden()]

    # Прогрев: первая загрузка модели в память занимает секунды и исказила бы
    # первый уровень. Кэши поиска (_corpus) тоже заполняются здесь.
    print(f"Прогрев... ({len(questions)} вопросов в прогоне)")
    answer(questions[0])

    rows = []
    for level in levels:
        row = asyncio.run(run_level(questions, level))
        rows.append(row)
        print(f"  одновременно {level}: {row['wall']:.1f} с", flush=True)

    base = rows[0]["wall"]
    print(f"\n{'одновр.':>8} {'прогон, с':>10} {'ускорение':>10} {'вопр/мин':>9} "
          f"{'ответ p50':>10} {'ответ max':>10} {'таймауты':>9}")
    for r in rows:
        print(f"{r['concurrency']:>8} {r['wall']:>10.1f} {base / r['wall']:>9.2f}x "
              f"{r['per_min']:>9.1f} {r['p50']:>9.1f}с {r['max']:>9.1f}с {r['timeouts']:>9}")


if __name__ == "__main__":
    main()
