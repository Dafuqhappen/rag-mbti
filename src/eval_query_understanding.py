"""Замер разборщика запросов отдельно от системы.

Запуск:  .venv/bin/python src/eval_query_understanding.py

Зачем отдельно: если структурные запросы находятся плохо, причин может быть
две — разборщик неправильно понял запрос, или фильтр правильный, но поиск
внутри отфильтрованного множества сработал плохо. Общая метрика их не
различает, поэтому компонент меряем сам по себе.

Метрики:
  exact  — разбор совпал с ожидаемым полностью (все четыре признака).
  partial— доля правильно определённых признаков.
  harmful— доля разборов, где хотя бы один признак ИНВЕРТИРОВАН. Это худший
           вид ошибки: фильтр не просто неточен, он исключает правильный ответ.
"""
from query_understanding import DICHOTOMIES, parse_filter

# Ожидаемые разборы. Признак отсутствует в словаре = его в запросе нет.
CASES = [
    ("интуитивный логик экстраверт рационал",
     {"is_extravert": True, "is_logic": True, "is_intuitive": True, "is_rational": True}),
    ("сенсорный этик интроверт иррационал",
     {"is_extravert": False, "is_logic": False, "is_intuitive": False, "is_rational": False}),
    ("интуитивный этик экстраверт иррационал",
     {"is_extravert": True, "is_logic": False, "is_intuitive": True, "is_rational": False}),
    ("какой тип логик и интроверт",
     {"is_logic": True, "is_extravert": False}),
    ("кто такой Дон Кихот", {}),
    ("у кого болевая этика отношений", {}),
    ("какие типы в квадре Дельта", {}),
    ("расскажи про Габена", {}),
]


def flatten(where: dict | None) -> dict:
    """Разворачивает where-фильтр Chroma обратно в плоский словарь признаков."""
    if not where:
        return {}
    if "$and" in where:
        return {k: v for clause in where["$and"] for k, v in clause.items()}
    return dict(where)


def main() -> None:
    exact = partial_scores = harmful = 0
    print(f"{'запрос':<40} {'вердикт'}")
    print("-" * 78)

    for query, expected in CASES:
        got = flatten(parse_filter(query))
        correct = sum(1 for key in DICHOTOMIES if got.get(key) == expected.get(key))
        inverted = [key for key in DICHOTOMIES
                    if key in expected and key in got and got[key] != expected[key]]

        exact += correct == len(DICHOTOMIES)
        partial_scores += correct / len(DICHOTOMIES)
        harmful += bool(inverted)

        if correct == len(DICHOTOMIES):
            verdict = "точно"
        elif inverted:
            verdict = f"ИНВЕРСИЯ: {', '.join(inverted)}"
        else:
            verdict = f"неполно ({correct}/{len(DICHOTOMIES)})"
        print(f"{query[:39]:<40} {verdict}")

    total = len(CASES)
    print()
    print(f"exact   = {exact / total:.2f}   ({exact}/{total})")
    print(f"partial = {partial_scores / total:.2f}")
    print(f"harmful = {harmful / total:.2f}   ({harmful}/{total}) <- инверсии, худшие ошибки")


if __name__ == "__main__":
    main()
