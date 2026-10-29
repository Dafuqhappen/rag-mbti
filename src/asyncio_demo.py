"""Демо asyncio: почему один блокирующий вызов останавливает весь бот.

Три пользователя пишут боту одновременно, ответ каждому готовится 2 секунды.
Запуск:  .venv/bin/python src/asyncio_demo.py

Три способа «подождать 2 секунды» внутри async-обработчика:
  blocking  — time.sleep: event loop стоит, остальные ждут своей очереди;
  awaiting  — await asyncio.sleep: loop свободен, пока ждём;
  to_thread — блокирующая функция в отдельном потоке (так сделано в bot.py).
"""
import asyncio
import time

USERS = ("Аня", "Борис", "Вера")
WORK_SEC = 2


def slow_answer_sync(user: str) -> str:
    """Синхронная функция, как answer() из rag.py: внутри ждёт сеть/модель."""
    time.sleep(WORK_SEC)
    return f"ответ для {user}"


async def handle_blocking(user: str, start: float) -> None:
    result = slow_answer_sync(user)          # loop заблокирован на 2 с
    print(f"  {time.perf_counter() - start:4.1f} с  {result}")


async def handle_awaiting(user: str, start: float) -> None:
    await asyncio.sleep(WORK_SEC)            # loop свободен, пока ждём
    print(f"  {time.perf_counter() - start:4.1f} с  ответ для {user}")


async def handle_to_thread(user: str, start: float) -> None:
    result = await asyncio.to_thread(slow_answer_sync, user)   # ждёт поток, loop свободен
    print(f"  {time.perf_counter() - start:4.1f} с  {result}")


async def serve_all(handler) -> float:
    """Как Telegram-библиотека: запускает обработчики всех входящих сообщений сразу."""
    start = time.perf_counter()
    await asyncio.gather(*(handler(user, start) for user in USERS))
    return time.perf_counter() - start


async def main() -> None:
    for label, handler in [("blocking  (time.sleep)", handle_blocking),
                           ("awaiting  (await asyncio.sleep)", handle_awaiting),
                           ("to_thread (как в bot.py)", handle_to_thread)]:
        print(f"{label}:")
        total = await serve_all(handler)
        print(f"  итого {total:.1f} с\n")


if __name__ == "__main__":
    asyncio.run(main())
