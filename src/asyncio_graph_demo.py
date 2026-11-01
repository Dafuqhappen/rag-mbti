"""Демо: граф управляет порядком шагов, но не тем, как шаг ждёт сети.

Запуск:  .venv/bin/python src/asyncio_graph_demo.py

Три пользователя одновременно, у каждого граф из одного шага «вызов модели» на
1 секунду. Рядом бежит «пульс» — корутина, которая каждые 0,1 с отмечается.
Если event loop заблокирован, пульс замирает: так видно, отвечал ли бот
остальным, пока шёл вызов.

Четыре варианта шага:
  sync_sleep         — обычная функция с time.sleep (синхронный клиент к LLM)
  async_blocking     — async def, но внутри тот же time.sleep (ошибка из мока)
  async_to_thread    — async def + asyncio.to_thread(time.sleep)
  async_sleep        — async def + asyncio.sleep (асинхронный клиент к LLM)
"""
import asyncio
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

CALL_SEC = 1.0
USERS = 3


class State(TypedDict):
    user: int


def sync_sleep(state: State) -> dict:
    time.sleep(CALL_SEC)
    return {}


async def async_blocking(state: State) -> dict:
    time.sleep(CALL_SEC)
    return {}


async def async_to_thread(state: State) -> dict:
    await asyncio.to_thread(time.sleep, CALL_SEC)
    return {}


async def async_sleep(state: State) -> dict:
    await asyncio.sleep(CALL_SEC)
    return {}


def build(step):
    graph = StateGraph(State)
    graph.add_node("call_llm", step)
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", END)
    return graph.compile()


async def measure(step) -> tuple[float, int]:
    app = build(step)
    beats = 0
    done = asyncio.Event()

    async def pulse():
        nonlocal beats
        while not done.is_set():
            await asyncio.sleep(0.1)
            beats += 1

    pulse_task = asyncio.create_task(pulse())
    start = time.perf_counter()
    await asyncio.gather(*(app.ainvoke({"user": u}) for u in range(USERS)))
    wall = time.perf_counter() - start
    done.set()
    await pulse_task
    return wall, beats


async def main() -> None:
    expected = int(CALL_SEC / 0.1)
    print(f"{USERS} пользователя, шаг «вызов модели» = {CALL_SEC:.0f} с; пульс живого loop ≈ {expected}+ отметок\n")
    print(f"{'шаг графа':<18} {'время':>7} {'пульс':>6}  вывод")
    for step in (sync_sleep, async_blocking, async_to_thread, async_sleep):
        wall, beats = await measure(step)
        verdict = "loop свободен" if beats >= expected * 0.7 else "loop ЗАБЛОКИРОВАН"
        print(f"{step.__name__:<18} {wall:>6.1f}с {beats:>6}  {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
