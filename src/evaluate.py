"""Оффлайн-замер качества ретривала на golden set.

Запуск:  .venv/bin/python src/evaluate.py                    # сравнить конфигурации
         .venv/bin/python src/evaluate.py semantic bge-m3    # подробно по одной

Метрики:
  Recall@k — какую долю правильных ответов достали (полнота).
  MRR      — 1 / позиция первого правильного (качество ранжирования).
Они дополняют друг друга: можно достать всё, но плохо отранжировать,
и наоборот — угадать первым, но не добрать остальные.
"""
import json
import sys
from collections import defaultdict

from config import DATA_DIR, DEFAULT_PROFILE, DEFAULT_GRANULARITY
from retrieve import search

K_VALUES = (3, 5)
METRICS = [f"recall@{k}" for k in K_VALUES] + ["mrr"]

# Что сравниваем: (метка, режим, профиль эмбеддингов, гранулярность, авто-фильтр).
# Меняем по одному измерению за раз, иначе непонятно, что именно дало эффект.
CONFIGURATIONS = [
    ("semantic/whole", "semantic", "bge-m3", "whole", False),
    ("hybrid/whole", "hybrid", "bge-m3", "whole", False),
    ("lexical/chunked", "lexical", "bge-m3", "chunked", False),
    ("hybrid/chunked", "hybrid", "bge-m3", "chunked", False),
    ("hybrid/chunked+filter", "hybrid", "bge-m3", "chunked", True),
]


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Доля ожидаемых ответов, попавших в первые k результатов."""
    hits = len(set(retrieved_ids[:k]) & set(expected_ids))
    return hits / len(expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1 / позиция первого правильного ответа; 0, если правильного нет вовсе."""
    for position, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            return 1 / position
    return 0.0


def load_golden() -> list[dict]:
    with open(DATA_DIR / "golden.json", encoding="utf-8") as f:
        return json.load(f)


def evaluate(mode: str, profile: str = DEFAULT_PROFILE,
             granularity: str = DEFAULT_GRANULARITY,
             auto_filter: bool = False) -> list[dict]:
    """Прогоняет весь golden set через одну конфигурацию поиска."""
    max_k = max(K_VALUES)
    rows = []
    for case in load_golden():
        hits = search(case["query"], k=max_k, mode=mode, profile=profile,
                      granularity=granularity, auto_filter=auto_filter)
        retrieved = [h["id"] for h in hits]
        rows.append({
            "query": case["query"],
            "tags": case["tags"],
            "expected": case["expected_ids"],
            "retrieved": retrieved,
            **{f"recall@{k}": recall_at_k(retrieved, case["expected_ids"], k) for k in K_VALUES},
            "mrr": reciprocal_rank(retrieved, case["expected_ids"]),
        })
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def by_tag(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        for tag in row["tags"]:
            grouped[tag].append(row)
    return grouped


def report_detailed(label: str, rows: list[dict]) -> None:
    print(f"КОНФИГУРАЦИЯ: {label}")
    print("=" * 78)
    for r in rows:
        mark = " " if r["mrr"] > 0 else "✗"
        print(f"{mark} {r['query'][:46]:<46} "
              + "  ".join(f"{m}={r[m]:.2f}" for m in METRICS))
    print()
    for tag, group in sorted(by_tag(rows).items()):
        line = "  ".join(f"{m}={mean([g[m] for g in group]):.2f}" for m in METRICS)
        print(f"{tag:<16} n={len(group):<3} {line}")


def report_comparison(results: dict[str, list[dict]]) -> None:
    """Таблица сравнения конфигураций — итог и разбивка по категориям."""
    labels = list(results)
    width = max(len(label) for label in labels) + 2
    tags = sorted({t for rows in results.values() for r in rows for t in r["tags"]})

    for metric in METRICS:
        print()
        print(f"{metric.upper():<18}" + "".join(f"{label:>{width}}" for label in labels))
        print("-" * (18 + width * len(labels)))
        for tag in tags:
            cells = "".join(
                f"{mean([r[metric] for r in by_tag(results[label])[tag]]):>{width}.2f}"
                for label in labels
            )
            print(f"{tag:<18}{cells}")
        total = "".join(
            f"{mean([r[metric] for r in results[label]]):>{width}.3f}" for label in labels
        )
        print(f"{'ИТОГО':<18}{total}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chosen_mode = sys.argv[1]
        chosen_profile = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROFILE
        chosen_gran = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_GRANULARITY
        chosen_filter = len(sys.argv) > 4 and sys.argv[4] == "filter"
        report_detailed(
            f"{chosen_mode}/{chosen_profile}/{chosen_gran}",
            evaluate(chosen_mode, chosen_profile, chosen_gran, chosen_filter),
        )
    else:
        report_comparison({
            label: evaluate(mode, profile, granularity, auto_filter)
            for label, mode, profile, granularity, auto_filter in CONFIGURATIONS
        })
