"""Слой поиска: семантический, лексический и гибридный.

Запуск:  .venv/bin/python src/retrieve.py "кто такой дон кихот"
         .venv/bin/python src/retrieve.py --mode lexical "кто такой Цезарь"
         .venv/bin/python src/retrieve.py --granularity chunked "болевая этика отношений"

При гранулярности chunked работает small-to-big: ранжирование идёт по мелким
чанкам (точность), а наружу отдаётся документ-родитель целиком (полнота
контекста для генерации).
"""
import sys
from collections import defaultdict
from functools import lru_cache

import chromadb

from config import (
    CHROMA_DIR, TOP_K, LEXICAL_WEIGHT, DEFAULT_PROFILE,
    DEFAULT_GRANULARITY, collection_name, semantic_weight,
)
from documents import parent_texts
from embeddings import embed
from lexical import BM25Index

RRF_K = 60             # сглаживающая константа RRF (стандартное значение)
FUSION_POOL = 10       # минимальный пул кандидатов от каждого ранкера
CHUNKS_PER_TYPE = 9    # описание + 8 функций: во столько раз шире надо брать пул


def get_collection(profile: str = DEFAULT_PROFILE,
                   granularity: str = DEFAULT_GRANULARITY):
    """Открывает коллекцию комбинации профиль x гранулярность."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(collection_name(profile, granularity))


@lru_cache(maxsize=None)
def _corpus(profile: str, granularity: str) -> tuple[dict[str, dict], BM25Index]:
    """Загружает коллекцию один раз: карта id -> запись и BM25-индекс.

    Кэшируем, потому что индекс строится по всем документам, и пересобирать
    его на каждый запрос — пустая трата. Для 16 документов это неважно,
    для 100k — принципиально.
    """
    raw = get_collection(profile, granularity).get(include=["documents", "metadatas"])
    by_id = {
        doc_id: {
            "id": doc_id,
            "parent_id": meta["parent_id"],
            "name": meta["name"],
            "slot": meta.get("slot", ""),
            "text": doc,
        }
        for doc_id, meta, doc in zip(raw["ids"], raw["metadatas"], raw["documents"])
    }
    return by_id, BM25Index(raw["ids"], raw["documents"])


def _allowed_ids(where: dict | None, profile: str, granularity: str) -> set[str] | None:
    """id документов, проходящих metadata-фильтр (None = фильтра нет)."""
    if where is None:
        return None
    return set(get_collection(profile, granularity).get(where=where)["ids"])


def search_semantic(query: str, k: int, where: dict | None = None,
                    profile: str = DEFAULT_PROFILE,
                    granularity: str = DEFAULT_GRANULARITY) -> list[dict]:
    """Векторный поиск: ближайшие соседи по эмбеддингу запроса.

    Если профиль считает векторы сам (ollama), передаём готовый вектор
    запроса; иначе отдаём текст, и Chroma эмбеддит его встроенной моделью.
    Важно: запрос и документы обязаны считаться ОДНОЙ моделью, иначе
    векторы окажутся из разных пространств и поиск потеряет смысл.
    """
    by_id, _ = _corpus(profile, granularity)
    vector = embed([query], profile)
    query_args = (
        {"query_embeddings": vector} if vector is not None else {"query_texts": [query]}
    )
    res = get_collection(profile, granularity).query(
        n_results=min(k, len(by_id)), where=where, **query_args
    )
    # API Chroma батчевый: принимает список запросов, поэтому ответ вложен
    # на один уровень — по элементу на запрос. Мы послали один, берём [0].
    return [by_id[doc_id] for doc_id in res["ids"][0]]


def search_lexical(query: str, k: int, where: dict | None = None,
                   profile: str = DEFAULT_PROFILE,
                   granularity: str = DEFAULT_GRANULARITY) -> list[dict]:
    """BM25-поиск по точным словам запроса (от эмбеддингов не зависит)."""
    by_id, index = _corpus(profile, granularity)
    allowed = _allowed_ids(where, profile, granularity)
    ranked = [
        doc_id for doc_id in index.top(query, k=len(by_id))
        if allowed is None or doc_id in allowed
    ]
    return [by_id[doc_id] for doc_id in ranked[:k]]


def search_hybrid(query: str, k: int, where: dict | None = None,
                  profile: str = DEFAULT_PROFILE,
                  granularity: str = DEFAULT_GRANULARITY,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """Гибрид: слияние семантики и BM25 через взвешенный Reciprocal Rank Fusion.

    RRF складывает не сами скоры (они в разных шкалах и несравнимы),
    а обратные позиции: 1/(RRF_K + позиция). Документ, попавший высоко
    хотя бы у одного ранкера, поднимается; попавший высоко у обоих —
    поднимается сильно. Никакой нормализации скоров не требуется.

    Веса нужны, когда ранкеры неравны по силе: при равных весах слабый
    ранкер проталкивает свои ошибки наверх и тянет качество вниз.
    """
    weights = weights or {
        "semantic": semantic_weight(profile, granularity),
        "lexical": LEXICAL_WEIGHT,
    }
    by_id, _ = _corpus(profile, granularity)
    pool = max(FUSION_POOL, k)
    rankings = {
        "semantic": search_semantic(query, pool, where, profile, granularity),
        "lexical": search_lexical(query, pool, where, profile, granularity),
    }

    fused: dict[str, float] = defaultdict(float)
    for ranker, hits in rankings.items():
        weight = weights.get(ranker, 1.0)
        for position, hit in enumerate(hits, start=1):
            fused[hit["id"]] += weight / (RRF_K + position)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [by_id[doc_id] for doc_id, _ in ordered[:k]]


MODES = {
    "semantic": search_semantic,
    "lexical": search_lexical,
    "hybrid": search_hybrid,
}


@lru_cache(maxsize=1)
def _parents() -> dict[str, str]:
    return parent_texts()


def _collapse_to_parents(hits: list[dict], k: int) -> list[dict]:
    """Small-to-big: схлопывает чанки в типы-родители, сохраняя порядок.

    Несколько чанков одного типа могут попасть в выдачу — засчитываем тип один
    раз, по его лучшему чанку, и отдаём наружу полный текст типа. Так поиск
    остаётся точным (ищем по мелкому), а генерация получает цельный контекст.
    """
    parents, seen, collapsed = _parents(), set(), []
    for hit in hits:
        parent_id = hit["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        collapsed.append({
            "id": parent_id,
            "name": hit["name"],
            "document": parents[parent_id],
            "matched_chunk": hit["id"],
            "matched_slot": hit["slot"],
            # Текст самого чанка тоже несём наружу: ретривал уже определил,
            # какой фрагмент отвечает на вопрос, и терять это знание перед
            # генерацией — значит заставлять модель искать заново.
            "matched_text": hit["text"],
        })
        if len(collapsed) == k:
            break
    return collapsed


def search(query: str, k: int = TOP_K, where: dict | None = None,
           mode: str = "hybrid", profile: str = DEFAULT_PROFILE,
           granularity: str = DEFAULT_GRANULARITY,
           weights: dict[str, float] | None = None,
           auto_filter: bool = False) -> list[dict]:
    """Единая точка входа. Всегда возвращает ТИПЫ, а не чанки.

    `mode` выбирает ранкер, `profile` — модель эмбеддингов, `granularity` —
    по чему индексируем, `auto_filter` — разбирать ли запрос моделью и строить
    metadata-фильтр автоматически. Всё вынесено в параметры намеренно: функция
    даёт механизм, а решение принимает вызывающий код.
    """
    if auto_filter and where is None:
        # Импорт внутри функции: разбор запроса нужен не всем вызовам, а
        # тянуть слой генерации при каждом импорте retrieve незачем.
        from query_understanding import parse_filter
        candidate = parse_filter(query)
        # Фильтр применяем, только если под него вообще что-то подходит.
        # Модель может собрать несуществующую комбинацию признаков, и тогда
        # выдача окажется пустой — это хуже, чем поиск без фильтра.
        if candidate and _allowed_ids(candidate, profile, granularity):
            where = candidate

    # При чанках один тип занимает до 9 позиций, поэтому пул берём шире —
    # иначе после схлопывания в родителей останется меньше k типов.
    pool = k if granularity == "whole" else k * CHUNKS_PER_TYPE
    kwargs = {"where": where, "profile": profile, "granularity": granularity}
    if mode == "hybrid":
        kwargs["weights"] = weights
    hits = MODES[mode](query, k=pool, **kwargs)
    return _collapse_to_parents(hits, k)


def main() -> None:
    args = sys.argv[1:]
    mode, profile, granularity = "hybrid", DEFAULT_PROFILE, DEFAULT_GRANULARITY
    while args and args[0].startswith("--"):
        flag, args = args[0], args[1:]
        if flag == "--mode":
            mode, args = args[0], args[1:]
        elif flag == "--profile":
            profile, args = args[0], args[1:]
        elif flag == "--granularity":
            granularity, args = args[0], args[1:]
    if not args:
        print('Использование: python src/retrieve.py [--mode M] [--profile P] '
              '[--granularity G] "вопрос"')
        raise SystemExit(1)

    query = " ".join(args)
    print(f"Запрос: {query}   ({mode}, {profile}, {granularity})\n")
    for i, h in enumerate(search(query, mode=mode, profile=profile, granularity=granularity), 1):
        matched = f"  <- совпал чанк: {h['matched_slot']}" if h["matched_slot"] else ""
        print(f"{i}. {h['name']}{matched}")


if __name__ == "__main__":
    main()
