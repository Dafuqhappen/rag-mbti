"""MCP-сервер поверх rag-mbti: справочник соционики как набор инструментов.

Запуск вручную не нужен — сервер запускает host (Claude Code) как дочерний
процесс и общается с ним через stdin/stdout строками JSON-RPC 2.0.

    claude mcp add rag-mbti -- /путь/к/.venv/bin/python /путь/к/src/mcp_server.py

ГЛАВНАЯ ЛОВУШКА stdio-сервера: stdout — это канал протокола. Любой print()
вклинится в поток JSON-RPC, и клиент получит мусор. Все логи — только в stderr.

Принцип отбора возможностей: сервер отдаёт то, чего у клиента НЕТ (доступ к
корпусу, точный гибридный поиск, детерминированный фильтр), и не отдаёт то, что
клиент делает лучше сам. Генерации ответа здесь нет: клиент — сильная модель,
рассуждает она, а сервер даёт ей проверенные данные.
"""
import logging
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from typing import Literal, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from documents import expand_function, load_json, whole_document
from embeddings import EmbeddingUnavailable
from lexical import tokenize
from query_understanding import DICHOTOMY_VOCABULARY
from retrieve import search

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-mbti")

server = MCPServer(
    name="rag-mbti",
    version="0.1.0",
    # Instructions читает модель клиента: это описание сервера в целом,
    # подсказка, когда и как им пользоваться.
    instructions=(
        "Справочник по соционике: 16 типов, Модель А из 8 функций, квадры и "
        "дихотомии. Используй эти инструменты вместо собственных знаний о "
        "соционике — данные проверены. Тип по названию — get_type; вопрос "
        "свободным текстом — search_types; отбор по признакам (экстраверт, логик, "
        "интуит, рационал) — filter_types."
    ),
)


# ---------------------------------------------------------------- данные

@lru_cache(maxsize=1)
def _types() -> dict[str, dict]:
    return {t["id"]: t for t in load_json("types.json")}


@lru_cache(maxsize=1)
def _glossary() -> dict:
    return load_json("functions.json")


def _is_code(alias: str) -> bool:
    """Аббревиатура вроде ИЛЭ или ENTP — у неё регистр несёт смысл."""
    return alias.isupper() and len(alias) <= 4


@lru_cache(maxsize=1)
def _alias_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Два индекса: основы слов (без учёта регистра) и коды (с учётом регистра).

    Основы: тот же стеммер, что в BM25 и в привратнике, и то же вычитание
    слов-дихотомий — основа "интуи" есть у восьми типов и никого не различает.

    Коды держим отдельно и сравниваем в исходном регистре: русский союз "или"
    иначе совпадает с кодом ИЛИ (Бальзак), и запрос "Дон Кихот или Габен"
    находит лишний тип. Аббревиатуру пишут заглавными, союз — строчными.
    Строчное "entp" при этом работает: у каждого типа есть вариант вида ENTp,
    который не аббревиатура и лежит в индексе основ.
    """
    noise = set(tokenize(DICHOTOMY_VOCABULARY))
    words: dict[str, set[str]] = defaultdict(set)
    codes: dict[str, set[str]] = defaultdict(set)
    for t in _types().values():
        for alias in [*t["aliases"], t["name"], t["id"]]:
            if _is_code(alias):
                codes[alias].add(t["id"])
                continue
            for stem in tokenize(alias):
                if stem not in noise:
                    words[stem].add(t["id"])
    return words, codes


def _resolve(name: str) -> list[str]:
    """Все типы, которые называет строка. 0, 1 или несколько кандидатов."""
    words, codes = _alias_index()
    found: set[str] = set()
    for raw in re.findall(r"\w+", name):
        found |= codes.get(raw, set())
    for stem in tokenize(name):
        found |= words.get(stem, set())
    return sorted(found)


class Dichotomies(TypedDict):
    extravert: bool
    logic: bool
    intuitive: bool
    rational: bool


class TypeCard(TypedDict):
    """Форма карточки типа.

    Не украшение: из этой аннотации SDK строит outputSchema, которую модель
    клиента видит в tools/list ещё до вызова. С голым dict схема пустая (null),
    и модель не знает заранее, какие поля придут.
    """
    id: str
    name: str
    aliases: list[str]
    quadra: str
    dichotomies: Dichotomies
    description: str
    functions: dict[str, str]


MBTI_TOKEN = re.compile(r"\b[A-Za-z]{4}\b")


def _mbti_code(t: dict) -> str | None:
    """Четырёхбуквенный код MBTI типа из его алиасов (ENTP, ISFJ...)."""
    return next((a for a in t["aliases"] if re.fullmatch(r"[A-Z]{4}", a)), None)


def _close_by_letters(name: str) -> list[str]:
    """«Did you mean»: типы, у которых совпадают 3 буквы MBTI из 4.

    ISTK -> ISTJ и ISTP. Сработает только на четырёхбуквенном латинском токене:
    для опечатки в имени ("Гексля") подсказывать по буквам бессмысленно.
    """
    for token in MBTI_TOKEN.findall(name):
        token = token.upper()
        close = [
            f"{t['name']} ({code})"
            for t in _types().values()
            if (code := _mbti_code(t)) and sum(a == b for a, b in zip(token, code)) == 3
        ]
        if close:
            return close
    return []


def _card(type_id: str) -> TypeCard:
    """Структурированная карточка типа: то, что модели удобно читать."""
    t = _types()[type_id]
    return {
        "id": t["id"],
        "name": t["name"],
        "aliases": t["aliases"],
        "quadra": t["quadra"],
        "dichotomies": {
            "extravert": t["is_extravert"],
            "logic": t["is_logic"],
            "intuitive": t["is_intuitive"],
            "rational": t["is_rational"],
        },
        "description": t["description"],
        "functions": {
            slot: expand_function(line, _glossary())
            for slot, line in t["functions"].items()
        },
    }


# ---------------------------------------------------------------- tools
# Docstring инструмента — это НЕ комментарий для разработчика. Его читает
# модель клиента, чтобы решить, звать ли инструмент и с какими аргументами.
# Фактически это промпт, и писать его надо как промпт.

class Hit(TypedDict):
    id: str
    name: str
    matched: str
    fragment: str


class SearchResult(TypedDict):
    """Конверт результата поиска.

    Флаг degraded — часть контракта, а не отладочная деталь: модель должна
    знать, что поиск упрощённый, иначе она примет слабую выдачу за полную и
    уверенно ответит «таких типов нет».
    """
    mode: str
    degraded: bool
    note: str
    hits: list[Hit]


# Предохранитель (circuit breaker). Таймаут запроса к ollama — 120 с: если хост
# не отказывает сразу, а виснет, каждый поиск ждал бы две минуты перед
# переходом на BM25. После сбоя полминуты даже не пытаемся. Состояние в памяти
# процесса годится для stdio-сервера (один процесс на клиента); у HTTP-сервера
# с несколькими воркерами его пришлось бы держать в общем хранилище.
BREAKER_COOLDOWN_SEC = 30
_semantic_down_until = 0.0

DEGRADED_NOTE = (
    "Смысловой поиск временно недоступен, работает поиск только по точным словам. "
    "Описательные запросы могут находиться хуже. Если выдача не подходит — "
    "переформулируй словами из названий типов и функций, либо используй "
    "get_type или filter_types."
)


def _envelope(hits: list[dict], degraded: bool) -> SearchResult:
    return {
        "mode": "lexical" if degraded else "hybrid",
        "degraded": degraded,
        "note": DEGRADED_NOTE if degraded else "",
        "hits": [
            {"id": h["id"], "name": h["name"],
             "matched": h["matched_slot"] or "profile", "fragment": h["matched_text"]}
            for h in hits
        ],
    }


@server.tool()
def search_types(query: str, k: int = 3) -> SearchResult:
    """Найти типы по вопросу на естественном языке.

    Гибридный поиск (смысловой + по точным словам) по описаниям типов и
    восьми функциям Модели А. Подходит для вопросов вроде «кто создаёт уют и
    избегает конфликтов» или «у кого болевая этика отношений».

    Возвращает до k типов с фрагментом, по которому тип был найден. Для полной
    карточки найденного типа вызови get_type. Если в ответе degraded=true, поиск
    работал в упрощённом режиме — прочитай note.
    """
    global _semantic_down_until
    if not 1 <= k <= 16:
        raise ToolError("k должен быть от 1 до 16 — всего типов шестнадцать.")

    if time.monotonic() >= _semantic_down_until:
        try:
            return _envelope(search(query, k=k), degraded=False)
        except EmbeddingUnavailable as exc:
            # Ловим ТОЛЬКО недоступность эмбеддингов. Любая другая ошибка должна
            # упасть громко, а не спрятаться за запасным путём.
            _semantic_down_until = time.monotonic() + BREAKER_COOLDOWN_SEC
            logger.warning("эмбеддинги недоступны, %d с работаем на BM25: %s",
                           BREAKER_COOLDOWN_SEC, exc)

    # BM25 не нужна ollama. Цена деградации известна из нашего же замера:
    # lexical/chunked даёт recall@3 0.932 (как гибрид), MRR 0.871 против 0.955.
    return _envelope(search(query, k=k, mode="lexical"), degraded=True)


@server.tool()
def filter_types(energy: Literal["extravert", "introvert"] | None = None,
                 judgment: Literal["logic", "ethics"] | None = None,
                 perception: Literal["intuitive", "sensing"] | None = None,
                 lifestyle: Literal["rational", "irrational"] | None = None) -> list[dict]:
    """Точно отобрать типы по дихотомиям. Не задавай признак, если он не важен.

    energy     — extravert (экстраверт) или introvert (интроверт)
    judgment   — logic (логик) или ethics (этик)
    perception — intuitive (интуит) или sensing (сенсорик)
    lifestyle  — rational (рационал) или irrational (иррационал)

    Буквы MBTI: E/I -> energy, N/S -> perception, T/F -> judgment,
    J -> lifestyle=rational, P -> lifestyle=irrational.
    Все четыре признака вместе однозначно задают ровно один тип.
    """
    # Параметры — слова, а не boolean: на «сенсорик» модель передавала
    # is_intuitive=true, отрицание ей даётся плохо (src/eval_tool_design.py:
    # exact 10/16 -> 15/16). Значение вне списка отклоняет сама схема: ToolError
    # со списком допустимых значений, а не молча расширенный фильтр.
    wanted = {feature: value == positive for feature, value, positive in (
        ("is_extravert", energy, "extravert"), ("is_logic", judgment, "logic"),
        ("is_intuitive", perception, "intuitive"), ("is_rational", lifestyle, "rational"),
    ) if value is not None}
    if not wanted:
        raise ToolError("Не задано ни одного признака. Для поиска по описанию "
                        "используй search_types.")
    return [
        {"id": t["id"], "name": t["name"], "quadra": t["quadra"]}
        for t in _types().values()
        if all(t[key] == value for key, value in wanted.items())
    ]


@server.tool()
def get_type(name: str) -> TypeCard:
    """Полная карточка типа по любому названию.

    Принимает псевдоним (Дон Кихот, Габен), код (ИЛЭ, СЛИ) или буквы MBTI
    (ENTP, ISTp); склонения допустимы. Возвращает квадру, дихотомии, описание
    и все восемь функций Модели А с позициями.
    """
    candidates = _resolve(name)

    # Ровно один — однозначно, отдаём карточку.
    if len(candidates) == 1:
        return _card(candidates[0])

    # Несколько — в строке названо несколько типов («Дон Кихот или Габен»).
    # Выбрать одного значит угадать за пользователя: дезинформация хуже
    # уточнения. Говорим модели, кто найден и как получить каждого.
    if candidates:
        names = ", ".join(_types()[c]["name"] for c in candidates)
        raise ToolError(
            f"В «{name}» названо несколько типов: {names}. "
            "Вызови get_type отдельно для каждого."
        )

    # Ноль. Текст ошибки — единственное, что увидит модель, поэтому он должен
    # вести к правильному следующему шагу и НЕ провоцировать выдумку: если
    # подсказать «попробуй filter_types» на «Илон Маск», модель сама присвоит
    # человеку признаки и получит тип, которого в базе нет.
    close = _close_by_letters(name)
    if close:
        raise ToolError(
            f"«{name}» не найден. Похоже на код MBTI с опечаткой, ближайшие "
            f"по буквам: {', '.join(close)}. Уточни у пользователя, какой имелся в виду."
        )
    raise ToolError(
        f"«{name}» не найден среди названий типов. Если это имя человека — "
        "типирования людей в базе нет, не определяй тип сам. Если пользователь "
        "описал тип словами — используй search_types, если перечислил признаки "
        "(экстраверт, логик...) — filter_types."
    )


# ---------------------------------------------------------------- resources
# Ресурс — не действие, а данные. Модель его не «вызывает»: приложение или
# пользователь прикрепляет его в контекст разговора.

@server.resource("mbti://types/{type_id}", mime_type="text/plain")
def type_document(type_id: str) -> str:
    """Полный текст описания одного типа."""
    if type_id not in _types():
        raise ToolError(f"Нет типа с id {type_id!r}.")
    return whole_document(_types()[type_id], _glossary())


@server.resource("mbti://glossary", mime_type="application/json")
def glossary() -> dict:
    """Названия восьми информационных аспектов: ЧИ, БЛ и так далее."""
    return _glossary()


# ---------------------------------------------------------------- prompts
# Промпт — заготовка, которую явно выбирает пользователь, как слэш-команду.

@server.prompt()
def compare_types(first: str, second: str) -> str:
    """Сравнить два типа по Модели А."""
    return (
        f"Сравни соционические типы «{first}» и «{second}». Сначала получи "
        "карточки обоих через get_type. Затем по каждой из восьми позиций Модели А "
        "покажи, какой аспект стоит у каждого типа, и сделай вывод, в чём они "
        "дополняют друг друга, а в чём будут конфликтовать. Опирайся только на "
        "данные из карточек."
    )


if __name__ == "__main__":
    server.run("stdio")
