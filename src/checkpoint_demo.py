"""Демо: что теряется при падении процесса и что спасает checkpointing.

Без LLM и без LangGraph — только механика. «Агент» делает ролик из 10 сцен,
каждая сцена — платный вызов внешнего сервиса. Процесс падает посреди сцены 7.

Запуск:
  .venv/bin/python src/checkpoint_demo.py memory --crash-at 7      # упал
  .venv/bin/python src/checkpoint_demo.py memory                   # перезапуск
  .venv/bin/python src/checkpoint_demo.py checkpoint --crash-at 7
  .venv/bin/python src/checkpoint_demo.py checkpoint
  .venv/bin/python src/checkpoint_demo.py reset                    # стереть всё

Два файла играют две разные роли:
  checkpoint_<thread>.json — НАША память: состояние агента на диске;
  billing.json             — ЧУЖОЙ мир: счёт сервиса генерации. Наш сбой его
                             не обнуляет, как не обнуляет счёт у настоящего провайдера.
"""
import json
import sys
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "demo_state"
THREAD_ID = "thread-42"
CHECKPOINT = STATE_DIR / f"checkpoint_{THREAD_ID}.json"
BILLING = STATE_DIR / "billing.json"
SCENES = 10
CREDITS_PER_SCENE = 5


def new_state() -> dict:
    """Состояние («карточка») с нуля: то, что агент знает о своей задаче."""
    return {
        "thread_id": THREAD_ID,
        "messages": [{"role": "user", "content": f"Сделай ролик из {SCENES} сцен"}],
        "done_scenes": [],
        "next_scene": 1,
    }


def charge(scene: int) -> None:
    """Внешний сервис списывает кредиты. Это происходит у НИХ, не у нас."""
    bill = json.loads(BILLING.read_text()) if BILLING.exists() else []
    bill.append({"scene": scene, "credits": CREDITS_PER_SCENE})
    BILLING.write_text(json.dumps(bill))


def generate_scene(state: dict, scene: int, crash_at: int | None) -> None:
    """Узел графа: один шаг агента."""
    charge(scene)  # запрос ушёл, деньги списаны
    if scene == crash_at:
        # Падаем ПОСЛЕ списания, но ДО того, как узнали результат:
        # самый неприятный момент — оплачено, а у нас нет ни видео, ни записи.
        print(f"  💥 сцена {scene}: кредиты списаны, сервер упал до ответа")
        raise SystemExit(1)
    state["messages"].append({"role": "tool", "content": f"сцена {scene}: video_{scene}.mp4"})
    state["done_scenes"].append(scene)
    state["next_scene"] = scene + 1
    print(f"  ✓ сцена {scene}")


def run(mode: str, crash_at: int | None) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    if mode == "checkpoint" and CHECKPOINT.exists():
        state = json.loads(CHECKPOINT.read_text())
        print(f"Нашёл контрольную точку {THREAD_ID}: продолжаю со сцены {state['next_scene']}")
    else:
        # Режим memory: state — обычная переменная в памяти процесса,
        # как messages внутри run_agent() в agent.py. Процесс умер — её нет.
        state = new_state()
        print("Состояния нет: начинаю со сцены 1")

    for scene in range(state["next_scene"], SCENES + 1):
        generate_scene(state, scene, crash_at)
        if mode == "checkpoint":
            # Вся разница двух режимов — эта строка: после каждого шага
            # карточка целиком уходит на диск.
            CHECKPOINT.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    bill = json.loads(BILLING.read_text())
    paid = [b["scene"] for b in bill]
    print(f"\nГотово. Оплачено сцен: {len(paid)} при нужных {SCENES} — "
          f"{len(paid) * CREDITS_PER_SCENE} кредитов вместо {SCENES * CREDITS_PER_SCENE}")
    print(f"Порядок оплат: {paid}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in ("memory", "checkpoint", "reset"):
        print(__doc__)
        raise SystemExit(1)
    if args[0] == "reset":
        for path in (CHECKPOINT, BILLING):
            path.unlink(missing_ok=True)
        print("Стёрто: контрольная точка и счёт")
        return
    crash_at = int(args[2]) if len(args) > 2 and args[1] == "--crash-at" else None
    run(args[0], crash_at)


if __name__ == "__main__":
    main()
