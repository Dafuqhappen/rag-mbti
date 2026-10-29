"""Замер качества генерации: заземление, ссылки, отказы.

Запуск:  .venv/bin/python src/eval_generation.py            # сравнить промпты
         .venv/bin/python src/eval_generation.py strict     # подробно по одному

Три метрики, каждая отвечает на свой вопрос:

  groundedness — не выдумала ли модель сущности, которых не было в контексте.
                 Это программный аналог faithfulness. Домен у нас конечный
                 (16 типов, 8 функций), поэтому проверяем точно, без LLM-судьи:
                 судья дороже, недетерминирован и сам может ошибаться.

  citation     — сослалась ли модель на источники. Без ссылок ответ RAG
                 нельзя проверить, и он теряет главное преимущество.

  refusal      — на вопросы, ответа на которые в корпусе НЕТ, модель обязана
                 сказать "не знаю". Это прямой замер того, работает ли
                 заземление: без него модель уверенно сочиняет.
"""
import json
import re
import sys

from config import DATA_DIR
from documents import parent_names
from prompts import SYSTEM_PROMPTS
from rag import answer

# Коды функций Модели А как отдельные токены
FUNCTION_RE = re.compile(r"\b(?:Ч|Б)(?:И|С|Л|Э)\b")

REFUSAL_MARKERS = ("нет данных", "не располагаю", "нет информации", "не могу ответить",
                   "отсутству", "не найдено", "нет сведений", "не указан")


def mentioned_entities(text: str, type_names: list[str]) -> set[str]:
    """Сущности домена, упомянутые в тексте: коды функций и названия типов."""
    found = set(FUNCTION_RE.findall(text))
    lowered = text.lower()
    found |= {name for name in type_names if name.lower() in lowered}
    return found


def groundedness(answer_text: str, context_text: str, type_names: list[str]) -> float | None:
    """Доля упомянутых сущностей, которые действительно есть в контексте.

    None, если модель вообще не назвала сущностей — тогда метрика неприменима,
    и усреднять по ней нельзя (иначе отказы завысят результат).
    """
    in_answer = mentioned_entities(answer_text, type_names)
    if not in_answer:
        return None
    in_context = mentioned_entities(context_text, type_names)
    return len(in_answer & in_context) / len(in_answer)


def has_citation(answer_text: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer_text))


def is_refusal(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate(prompt: str) -> list[dict]:
    with open(DATA_DIR / "generation_eval.json", encoding="utf-8") as f:
        cases = json.load(f)
    type_names = list(parent_names().values())

    rows = []
    for case in cases:
        result = answer(case["question"], prompt=prompt)
        context_text = "\n".join(h["document"] for h in result.get("context", []))
        rows.append({
            "question": case["question"],
            "answerable": case["answerable"],
            "answer": result["answer"],
            "groundedness": groundedness(result["answer"], context_text, type_names),
            "citation": has_citation(result["answer"]),
            "refusal": is_refusal(result["answer"]),
            "elapsed_sec": result["elapsed_sec"],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    grounded = [r["groundedness"] for r in rows if r["groundedness"] is not None]

    def share(items, key):
        return sum(bool(i[key]) for i in items) / len(items) if items else 0.0

    return {
        "groundedness": sum(grounded) / len(grounded) if grounded else 0.0,
        "citation (отвечаемые)": share(answerable, "citation"),
        "refusal (неотвечаемые)": share(unanswerable, "refusal"),
        # Ложный отказ: в корпусе ответ есть, а модель сказала "не знаю"
        "ложных отказов": share(answerable, "refusal"),
        "сек/ответ": sum(r["elapsed_sec"] for r in rows) / len(rows),
    }


def report_detailed(prompt: str, rows: list[dict]) -> None:
    print(f"ПРОМПТ: {prompt}")
    print("=" * 78)
    for r in rows:
        kind = "отвечаемый" if r["answerable"] else "НЕотвечаемый"
        ok = "отказ" if r["refusal"] else f"ground={r['groundedness']}"
        print(f"\n[{kind}] {r['question']}")
        print(f"  {ok}, ссылки={r['citation']}, {r['elapsed_sec']:.1f}с")
        print(f"  {r['answer'][:260]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report_detailed(sys.argv[1], evaluate(sys.argv[1]))
    else:
        results = {name: summarize(evaluate(name)) for name in SYSTEM_PROMPTS}
        metrics = list(next(iter(results.values())))
        names = list(results)
        print(f"{'':<26}" + "".join(f"{n:>12}" for n in names))
        print("-" * (26 + 12 * len(names)))
        for metric in metrics:
            print(f"{metric:<26}" + "".join(f"{results[n][metric]:>12.2f}" for n in names))
