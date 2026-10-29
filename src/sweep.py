"""Развертка по весам гибридного слияния (hyperparameter sweep).

Запуск:  .venv/bin/python src/sweep.py [профиль] [гранулярность]

Вес лексического ранкера держим равным 1.0 и меняем вес семантического
от 0 (чистый BM25) до 2 (семантика вдвое важнее). Смотрим, где метрики
достигают максимума, вместо того чтобы подбирать вес на глаз.

Развертку нужно делать ЗАНОВО для каждой конфигурации: оптимальный вес
зависит и от силы модели, и от гранулярности индекса. Переиспользовать вес,
подобранный на другой конфигурации, — значит сравнивать их нечестно.
"""
import sys

from config import DEFAULT_PROFILE, DEFAULT_GRANULARITY
from evaluate import K_VALUES, METRICS, load_golden, mean, recall_at_k, reciprocal_rank
from retrieve import search

SEMANTIC_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]


def evaluate_weights(semantic_weight: float, profile: str,
                     granularity: str) -> dict[str, float]:
    """Прогоняет golden set через гибрид с заданным весом семантики."""
    weights = {"semantic": semantic_weight, "lexical": 1.0}
    max_k = max(K_VALUES)
    rows = []
    for case in load_golden():
        hits = search(case["query"], k=max_k, mode="hybrid", profile=profile,
                      granularity=granularity, weights=weights)
        retrieved = [h["id"] for h in hits]
        rows.append({
            **{f"recall@{k}": recall_at_k(retrieved, case["expected_ids"], k) for k in K_VALUES},
            "mrr": reciprocal_rank(retrieved, case["expected_ids"]),
        })
    return {m: mean([r[m] for r in rows]) for m in METRICS}


if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROFILE
    granularity = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GRANULARITY
    print(f"конфигурация: {profile} / {granularity}\n")
    print(f"{'вес семантики':<16}" + "".join(f"{m:>12}" for m in METRICS))
    print("-" * (16 + 12 * len(METRICS)))

    results = {}
    for weight in SEMANTIC_WEIGHTS:
        scores = results[weight] = evaluate_weights(weight, profile, granularity)
        print(f"{weight:<16.1f}" + "".join(f"{scores[m]:>12.3f}" for m in METRICS))

    print()
    for metric in METRICS:
        best = max(results, key=lambda w: results[w][metric])
        print(f"лучший вес по {metric:<10} = {best:.1f}  ({results[best][metric]:.3f})")
