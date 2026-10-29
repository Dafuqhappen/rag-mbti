"""Замер: влияет ли ФОРМА параметров инструмента на ошибки модели.

Запуск:  .venv/bin/python src/eval_tool_design.py           # 4 варианта, порядок как на сервере
         .venv/bin/python src/eval_tool_design.py --orders  # bool и server на всех 6 порядках

Гипотеза: на «сенсорик» qwen передаёт `is_intuitive=true`, потому что boolean
требует отрицания («сенсорик» = НЕ интуит). Enum без отрицания уберёт ошибку.

Меряем один компонент — первый вызов модели, инструмент не исполняется. Так
результат зависит только от интерфейса, а не от поиска или генерации ответа.
Варианты filter_types несут одну и ту же информацию, меняется только форма:
  bool    — как было на сервере до 27.10 (is_intuitive: true/false), схема заморожена;
  enum_en — perception: "intuitive" | "sensing";
  enum_ru — perception: "интуит" | "сенсорик" (слово можно переписать из вопроса);
  server  — то, что сервер отдаёт сейчас: enum_en в обёртке Pydantic
            (anyOf: [enum, null]). Форма другая, поэтому мерим отдельно.

ПОРЯДОК ИНСТРУМЕНТОВ — скрытая переменная. Первый замер ставил boolean-вариант
вторым (как на сервере), а enum — последним; перестановка одного boolean сдвинула
exact с 10/16 до 13/16. Поэтому варианты сравниваются при одном порядке, а шум
оценивается прогоном на всех перестановках (--orders).

Метрики (как в eval_query_understanding.py):
  filter  — первым вызван filter_types;
  exact   — разбор совпал полностью: нужные признаки верны, лишних нет;
  feature — доля верно переданных признаков из ожидаемых;
  harmful — фильтр исключает правильный ответ: признак ИНВЕРТИРОВАН или
            добавлен ЛИШНИЙ (в вопросе «логики-рационалы» нет «экстраверт»);
  invalid — значение вне enum (ollama не проверяет схему, проверяем сами).
"""
import sys
from itertools import permutations

from agent import SYSTEM, connect, llm

FEATURES = ("is_extravert", "is_logic", "is_intuitive", "is_rational")

# Признак -> (имя enum-параметра, значение для true, значение для false)
ENUM_EN = {"is_extravert": ("energy", "extravert", "introvert"),
           "is_logic": ("judgment", "logic", "ethics"),
           "is_intuitive": ("perception", "intuitive", "sensing"),
           "is_rational": ("lifestyle", "rational", "irrational")}
ENUM_RU = {"is_extravert": ("energy", "экстраверт", "интроверт"),
           "is_logic": ("judgment", "логик", "этик"),
           "is_intuitive": ("perception", "интуит", "сенсорик"),
           "is_rational": ("lifestyle", "рационал", "иррационал")}

# Описание то же, что у серверного filter_types, только под другую форму.
ENUM_DESCRIPTION = """Точно отобрать типы по дихотомиям. Не задавай признак, если он не важен.

energy     — {is_extravert[1]} (экстраверт) или {is_extravert[2]} (интроверт)
judgment   — {is_logic[1]} (логик) или {is_logic[2]} (этик)
perception — {is_intuitive[1]} (интуит) или {is_intuitive[2]} (сенсорик)
lifestyle  — {is_rational[1]} (рационал) или {is_rational[2]} (иррационал)

Буквы MBTI: E/I -> energy, N/S -> perception, T/F -> judgment,
J -> lifestyle={is_rational[1]}, P -> lifestyle={is_rational[2]}.
Все четыре признака вместе однозначно задают ровно один тип."""

# Вопрос -> ожидаемый разбор. Признака нет в словаре = его в вопросе нет.
T, F = True, False
CASES = [
    ("Какой тип: интроверт, логик, сенсорик, иррационал?",
     {"is_extravert": F, "is_logic": T, "is_intuitive": F, "is_rational": F}),
    ("Найди тип: экстраверт, этик, сенсорик, рационал.",
     {"is_extravert": T, "is_logic": F, "is_intuitive": F, "is_rational": T}),
    ("Кто из типов интроверт, этик, интуит и иррационал?",
     {"is_extravert": F, "is_logic": F, "is_intuitive": T, "is_rational": F}),
    ("Подбери тип: сенсорный логик-экстраверт, иррационал.",
     {"is_extravert": T, "is_logic": T, "is_intuitive": F, "is_rational": F}),
    ("Интровертный сенсорный этик-рационал — это кто?",
     {"is_extravert": F, "is_logic": F, "is_intuitive": F, "is_rational": T}),
    ("Какой тип: экстраверт, логик, интуит, рационал?",
     {"is_extravert": T, "is_logic": T, "is_intuitive": T, "is_rational": T}),
    ("Нужен тип: не логик, не интуит, экстраверт.",
     {"is_extravert": T, "is_logic": F, "is_intuitive": F}),
    ("Какие типы — этики-интроверты?", {"is_extravert": F, "is_logic": F}),
    ("Покажи всех сенсориков-иррационалов.", {"is_intuitive": F, "is_rational": F}),
    ("Какие есть экстравертные интуиты?", {"is_extravert": T, "is_intuitive": T}),
    ("Перечисли логиков-рационалов.", {"is_logic": T, "is_rational": T}),
    ("Интроверт и сенсорик — какие это типы?", {"is_extravert": F, "is_intuitive": F}),
    ("Кто иррациональный этик?", {"is_rational": F, "is_logic": F}),
    ("Какие типы сенсорики?", {"is_intuitive": F}),
    ("Какие типы интроверты?", {"is_extravert": F}),
    ("Какие типы иррационалы?", {"is_rational": F}),
]


# Схема boolean-варианта, как её отдавал сервер до перехода на слова (27.10).
BOOL_DESCRIPTION = """Точно отобрать типы по дихотомиям. Не задавай признак, если он не важен.

is_extravert — экстраверт (true) или интроверт (false)
is_logic     — логик (true) или этик (false)
is_intuitive — интуит (true) или сенсорик (false)
is_rational  — рационал (true) или иррационал (false)

Буквы MBTI: E/I -> is_extravert, N/S -> is_intuitive, T/F -> is_logic,
J -> is_rational=true, P -> is_rational=false.
Все четыре признака вместе однозначно задают ровно один тип.
"""
BOOL_TOOL = {"type": "function", "function": {
    "name": "filter_types",
    "description": BOOL_DESCRIPTION,
    "parameters": {"type": "object", "title": "filter_typesArguments", "properties": {
        feature: {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None,
                  "title": feature.replace("_", " ").title()}
        for feature in FEATURES}},
}}


def enum_tool(mapping: dict) -> dict:
    properties = {param: {"type": "string", "enum": [yes, no]}
                  for param, yes, no in mapping.values()}
    return {"type": "function", "function": {
        "name": "filter_types",
        "description": ENUM_DESCRIPTION.format(**mapping),
        "parameters": {"type": "object", "properties": properties},
    }}


def decode(args: dict, mapping: dict | None) -> tuple[dict, int]:
    """Аргументы вызова -> признаки-boolean и число недопустимых значений."""
    if mapping is None:
        return {k: v for k, v in args.items() if k in FEATURES and isinstance(v, bool)}, 0
    decoded, invalid = {}, 0
    for feature, (param, yes, no) in mapping.items():
        if param not in args or args[param] is None:
            continue
        value = str(args[param]).strip().lower()
        if value == yes:
            decoded[feature] = True
        elif value == no:
            decoded[feature] = False
        else:
            invalid += 1
    return decoded, invalid


def score(variant: str, tools: list[dict], mapping: dict | None) -> dict:
    totals = {"filter": 0, "exact": 0, "harmful": 0, "inverted": 0, "extra": 0,
              "invalid": 0, "right": 0, "expected": 0}
    misses = []
    for question, expected in CASES:
        message = llm([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": question}], tools)
        calls = message.get("tool_calls") or []
        totals["expected"] += len(expected)
        if not calls or calls[0]["function"]["name"] != "filter_types":
            first = calls[0]["function"]["name"] if calls else "без вызова"
            misses.append(f"{question} → {first}")
            continue
        got, invalid = decode(calls[0]["function"].get("arguments") or {}, mapping)
        totals["filter"] += 1
        totals["invalid"] += invalid
        totals["right"] += sum(got.get(k) == v for k, v in expected.items())
        inverted = [k for k, v in expected.items() if k in got and got[k] != v]
        extra = [k for k in got if k not in expected]
        totals["inverted"] += bool(inverted)
        totals["extra"] += bool(extra)
        totals["harmful"] += bool(inverted or extra)
        if got == expected and not invalid:
            totals["exact"] += 1
        else:
            misses.append(f"{question} → {calls[0]['function'].get('arguments')}"
                          + (f"  ПЕРЕВЁРНУТО: {', '.join(inverted)}" if inverted else "")
                          + (f"  ЛИШНЕЕ: {', '.join(extra)}" if extra else ""))
    print(f"\n[{variant}] промахи:")
    for m in misses or ["нет"]:
        print(f"  {m}")
    return totals


def in_order(server_tools: list[dict], filter_tool: dict, order: tuple[str, ...]) -> list[dict]:
    by_name = {t["function"]["name"]: t for t in server_tools} | {"filter_types": filter_tool}
    return [by_name[name] for name in order]


def print_table(rows: dict[str, dict]) -> None:
    n = len(CASES)
    print(f"\n{'вариант':>22} {'filter':>7} {'exact':>7} {'feature':>8} {'harmful':>8} "
          f"{'перевёрн.':>9} {'лишний':>7} {'invalid':>8}")
    for name, r in rows.items():
        print(f"{name:>22} {r['filter']:>4}/{n} {r['exact']:>4}/{n} "
              f"{r['right'] / r['expected']:>8.0%} {r['harmful']:>5}/{n} "
              f"{r['inverted']:>6}/{n} {r['extra']:>4}/{n} {r['invalid']:>8}")


def main() -> None:
    client, server_tools = connect()
    client.close()  # нужны только схемы инструментов; исполнять их не будем
    server_order = tuple(t["function"]["name"] for t in server_tools)
    server_filter = next(t for t in server_tools if t["function"]["name"] == "filter_types")
    variants = {
        "bool": (BOOL_TOOL, None),
        "enum_en": (enum_tool(ENUM_EN), ENUM_EN),
        "enum_ru": (enum_tool(ENUM_RU), ENUM_RU),
        "server": (server_filter, ENUM_EN),
    }
    if "--orders" in sys.argv:
        orders = list(permutations(server_order))
        jobs = {f"{name} {'>'.join(o[:1] for o in order)}": (tool, mapping, order)
                for name in ("bool", "server") for order in orders
                for tool, mapping in [variants[name]]}
    else:
        jobs = {name: (tool, mapping, server_order) for name, (tool, mapping) in variants.items()}
    rows = {label: score(label, in_order(server_tools, tool, order), mapping)
            for label, (tool, mapping, order) in jobs.items()}
    print_table(rows)
    if "--orders" in sys.argv:
        for name in ("bool", "server"):
            exact = [r["exact"] for label, r in rows.items() if label.startswith(name + " ")]
            harmful = [r["harmful"] for label, r in rows.items() if label.startswith(name + " ")]
            print(f"{name:>8}: exact {min(exact)}–{max(exact)} (среднее {sum(exact) / len(exact):.1f}) "
                  f"из {len(CASES)}, harmful {min(harmful)}–{max(harmful)}")


if __name__ == "__main__":
    main()
