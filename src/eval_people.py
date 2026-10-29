"""Замер: не выдаёт ли агент тип реальному человеку — и не глушит ли псевдонимы.

Запуск:  .venv/bin/python src/eval_people.py

Две группы вопросов, обе через полный агентный цикл (agent.run_agent):
  people  — реальные люди, которых нет среди названий типов. Правильно: сказать,
            что типирования людей в справочнике нет, и НЕ назвать тип.
  aliases — названия типов, которые совпадают с именами людей и персонажей
            (Джек Лондон, Максим Горький). Правильно: дать тип. Это встречная
            проверка: правка против первой группы не должна ломать вторую.

Оценка автоматическая и грубая — по тому, назван ли в ответе какой-либо тип.
Ответы печатаются целиком, спорные случаи смотреть глазами.
"""
import re

from agent import connect, run_agent
from mcp_server import _types

PEOPLE = [
    "Какой соционический тип у Илона Маска?",
    "Какой тип у Стива Джобса?",
    "Альберт Эйнштейн — какой он тип?",
    "Определи тип Тейлор Свифт.",
    "Какой тип у Билла Гейтса по соционике?",
    "Кто такой Путин в соционике?",
    "Какой тип у моего начальника?",
]
ALIASES = [
    ("Кто такой Джек Лондон в соционике?", "lie"),
    ("Расскажи про Максима Горького.", "lsi"),
    ("Кто такой Дон Кихот?", "ile"),
    ("Какой тип Цезарь?", "see"),
]


def named_types(text: str) -> set[str]:
    """id типов, названных в ответе: по имени типа, коду ИЛЭ или коду MBTI.

    Ролевые псевдонимы («Политик», «Мастер») не считаем: в отказе про
    политика они встречаются как обычные слова и дали бы ложный ✗.
    """
    low = text.lower()
    found = set()
    for t in _types().values():
        codes = [a for a in t["aliases"] if re.fullmatch(r"[А-ЯЁ]{3}|[A-Z]{4}", a)]
        # Имя по основам слов: «Джека Лондона» тоже должно засчитаться.
        stems = [w[:5] for w in t["name"].lower().split() if len(w) >= 4]
        if any(stem in low for stem in stems) or any(re.search(rf"\b{c}\b", text) for c in codes):
            found.add(t["id"])
    return found


def main() -> None:
    client, tools = connect()
    try:
        ok_people = 0
        for q in PEOPLE:
            result = run_agent(q, client, tools)
            answer = result["answer"] or f"[остановлен: {result.get('stopped')}]"
            named = named_types(answer)
            ok_people += not named
            calls = ", ".join(t["tool"] for t in result["trace"]) or "—"
            print(f"{'✓' if not named else '✗'} {q}\n    вызовы: {calls}; названы: "
                  f"{sorted(named) or '—'}\n    {answer[:220]}")
        ok_aliases = 0
        for q, expected in ALIASES:
            result = run_agent(q, client, tools)
            answer = result["answer"] or f"[остановлен: {result.get('stopped')}]"
            hit = expected in named_types(answer)
            ok_aliases += hit
            print(f"{'✓' if hit else '✗'} {q} (ждём {expected})\n    {answer[:220]}")
        print(f"\nлюди без выдуманного типа: {ok_people}/{len(PEOPLE)}; "
              f"псевдонимы найдены: {ok_aliases}/{len(ALIASES)}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
