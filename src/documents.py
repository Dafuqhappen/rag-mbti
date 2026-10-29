"""Сборка документов из корпуса: цельные типы и мелкие чанки.

Здесь живёт вся логика «как из данных сделать текст для поиска». Вынесена
отдельно, потому что нужна двум сторонам: загрузчику (что индексировать)
и поиску (что вернуть как документ-родитель).

Две гранулярности:
  whole   — один документ на тип. Хорош для вопросов «кто такой X».
  chunked — описание и каждая функция отдельно (9 чанков на тип). Хорош для
            вопросов про конкретный слот: в мелком куске «болевая» и «БЭ»
            стоят рядом и не тонут среди остальных семи функций.
"""
import json

from config import DATA_DIR


def load_json(filename: str):
    """Читает JSON из data/ с явной кодировкой."""
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def expand_function(line: str, glossary: dict) -> str:
    """Подставляет полное и альтернативное название функции.

    В корпусе хранится только код ("ЧИ — видит возможности..."), названия живут
    в глоссарии — единая точка правды. Здесь они склеиваются обратно, чтобы
    попасть в эмбеддинг: запрос "у кого болевая этика отношений" должен
    находить документ, где написано только "БЭ".
    """
    code = line[:2]
    entry = glossary.get(code)
    if not entry:
        return line
    rest = line[2:].lstrip(" —").strip()
    return f"{code} ({entry['name']}, {entry['alt']}) — {rest}"


def _dichotomies(t: dict) -> dict:
    return {
        "is_extravert": t["is_extravert"],
        "is_logic": t["is_logic"],
        "is_intuitive": t["is_intuitive"],
        "is_rational": t["is_rational"],
    }


def whole_document(t: dict, glossary: dict) -> str:
    """Тип целиком одним текстовым блоком."""
    functions = "\n".join(
        f"{slot}: {expand_function(line, glossary)}"
        for slot, line in t["functions"].items()
    )
    return (
        f"Тип: {t['name']}\n"
        f"Другие названия: {', '.join(t['aliases'])}\n"
        f"Квадра: {t['quadra']}\n"
        f"Описание: {t['description']}\n"
        f"Функции (Модель А):\n{functions}"
    )


def build_whole(types: list[dict], glossary: dict) -> list[dict]:
    """Гранулярность whole: по документу на тип."""
    return [
        {
            "id": t["id"],
            "text": whole_document(t, glossary),
            "metadata": {"parent_id": t["id"], "name": t["name"],
                         "quadra": t["quadra"], "slot": "", **_dichotomies(t)},
        }
        for t in types
    ]


def build_chunks(types: list[dict], glossary: dict) -> list[dict]:
    """Гранулярность chunked: описание + каждая функция отдельным чанком.

    В каждый чанк дублируем имя и синонимы типа. Без этого чанк функции
    перестаёт быть привязанным к типу: поиск нашёл бы «болевую БЭ», но не смог
    бы понять, чья она. Небольшое дублирование здесь окупается точностью.
    """
    chunks = []
    for t in types:
        header = f"Тип: {t['name']} ({', '.join(t['aliases'])})"
        base_meta = {"parent_id": t["id"], "name": t["name"],
                     "quadra": t["quadra"], **_dichotomies(t)}

        chunks.append({
            "id": f"{t['id']}#profile",
            "text": f"{header}\nКвадра: {t['quadra']}\nОписание: {t['description']}",
            "metadata": {**base_meta, "slot": "profile"},
        })
        for slot, line in t["functions"].items():
            # "4_болевая" -> "болевая": позиция важна для поиска, номер — нет
            slot_name = slot.split("_", 1)[1]
            chunks.append({
                "id": f"{t['id']}#{slot}",
                "text": (f"{header}\n{slot_name} функция (позиция {slot.split('_')[0]} "
                         f"в Модели А): {expand_function(line, glossary)}"),
                "metadata": {**base_meta, "slot": slot_name},
            })
    return chunks


BUILDERS = {"whole": build_whole, "chunked": build_chunks}


def build(granularity: str) -> list[dict]:
    """Собирает документы выбранной гранулярности."""
    types, glossary = load_json("types.json"), load_json("functions.json")
    return BUILDERS[granularity](types, glossary)


def parent_texts() -> dict[str, str]:
    """id типа -> его полный текст. Нужен для small-to-big: ищем по чанкам,
    а в генерацию отдаём цельный документ-родитель."""
    types, glossary = load_json("types.json"), load_json("functions.json")
    return {t["id"]: whole_document(t, glossary) for t in types}


def parent_names() -> dict[str, str]:
    return {t["id"]: t["name"] for t in load_json("types.json")}
