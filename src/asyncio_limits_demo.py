"""Демо: параллельные платные задачи — скорость, лимиты, бюджет.

Сервис генерации (выдуманный, как типичный платный API видео): одна сцена = 1 с и
5 кредитов, одновременно не больше 3 задач, лишние отклоняет с ошибкой 429.
Кредиты списываются в момент приёма задачи.

Запуск:  .venv/bin/python src/asyncio_limits_demo.py
"""
import asyncio
import time

SCENES = range(1, 11)
COST = 5
SERVICE_LIMIT = 3     # сколько задач сервис держит одновременно
SCENE_SEC = 1.0


class GenerationService:
    def __init__(self, balance: int):
        self.balance = balance
        self.active = 0
        self.charged: list[int] = []     # за что списали
        self.delivered: list[int] = []   # что мы реально получили

    async def generate(self, scene: int) -> str:
        if self.active >= SERVICE_LIMIT:
            raise RuntimeError(f"429: сцена {scene} — слишком много задач одновременно")
        if self.balance < COST:
            raise RuntimeError(f"402: сцена {scene} — не хватает кредитов")
        self.balance -= COST
        self.charged.append(scene)
        self.active += 1
        try:
            await asyncio.sleep(SCENE_SEC)
            self.delivered.append(scene)
            return f"video_{scene}.mp4"
        finally:
            self.active -= 1


async def sequential(svc: GenerationService) -> None:
    """По одной: надёжно, но медленно."""
    for scene in SCENES:
        await svc.generate(scene)


async def naive_gather(svc: GenerationService) -> None:
    """Всё сразу, без лимита и без проверки бюджета."""
    try:
        await asyncio.gather(*(svc.generate(s) for s in SCENES))
    except RuntimeError as exc:
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        print(f"  gather вернул первую ошибку: {exc}")
        print(f"  ...а задач ещё работает в фоне: {len(others)} — их никто не отменил")
        # Выходим из функции. asyncio.run при завершении отменит оставшиеся
        # задачи: кредиты за них уже списаны, видео не будет.


async def careful(svc: GenerationService) -> None:
    """Бюджет — до старта, одновременность — семафором, ошибки — группой."""
    quote = len(SCENES) * COST
    if quote > svc.balance:
        print(f"  смета {quote} кредитов > баланс {svc.balance}: не запускаю ничего")
        return
    gate = asyncio.Semaphore(SERVICE_LIMIT)

    async def one(scene: int) -> str:
        async with gate:                 # не больше SERVICE_LIMIT задач в полёте
            return await svc.generate(scene)

    # TaskGroup (Python 3.11+): если одна задача упала, остальные ОТМЕНЯЮТСЯ,
    # а не продолжают молча тратить деньги, как при gather.
    async with asyncio.TaskGroup() as group:
        for scene in SCENES:
            group.create_task(one(scene))


def run(label: str, scenario, balance: int) -> None:
    svc = GenerationService(balance)
    print(f"{label} (баланс {balance}):")
    start = time.perf_counter()
    asyncio.run(scenario(svc))
    took = time.perf_counter() - start
    print(f"  время {took:.1f} с | списано за {len(svc.charged)} сцен "
          f"({len(svc.charged) * COST} кр.) | получено видео: {len(svc.delivered)}\n")


if __name__ == "__main__":
    run("1. По одной в цикле for", sequential, balance=100)
    run("2. gather без лимита", naive_gather, balance=100)
    run("3. Семафор + TaskGroup", careful, balance=100)
    run("4. Семафор + TaskGroup, денег мало", careful, balance=25)
