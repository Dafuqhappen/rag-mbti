"""Конвейер загрузки: корпус -> документы -> эмбеддинги -> Chroma.

Запуск:  .venv/bin/python src/ingest.py                  # профиль и гранулярность по умолчанию
         .venv/bin/python src/ingest.py bge-m3 chunked   # конкретная комбинация
         .venv/bin/python src/ingest.py all              # все комбинации

Идемпотентен: повторный запуск пересобирает коллекцию с нуля, дублей не будет.
"""
import sys

import chromadb

from config import (
    CHROMA_DIR, EMBEDDING_PROFILES, DEFAULT_PROFILE,
    GRANULARITIES, DEFAULT_GRANULARITY, collection_name,
)
from documents import build
from embeddings import BATCH_TIMEOUT_SEC, embed


def build_collection(profile: str, granularity: str) -> None:
    """Пересобирает коллекцию для одной комбинации профиль x гранулярность."""
    docs = build(granularity)
    name = collection_name(profile, granularity)

    # Явный шаг: текст -> векторы. Для профиля chroma-default вернётся None,
    # и векторы посчитает сама Chroma внутри add().
    vectors = embed([d["text"] for d in docs], profile, timeout=BATCH_TIMEOUT_SEC)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.create_collection(name)

    # Один батчевый вызов вместо add() в цикле
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        # В metadata только скаляры. Дихотомии здесь — чтобы фильтровать
        # структурные запросы через where, а не решать их семантикой.
        # parent_id связывает чанк с типом-родителем (small-to-big).
        metadatas=[d["metadata"] for d in docs],
        **({"embeddings": vectors} if vectors is not None else {}),
    )

    dims = len(vectors[0]) if vectors else "определяет Chroma"
    print(f"[{profile}/{granularity}] документов: {len(docs)} -> '{name}', размерность: {dims}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "all":
        combos = [(p, g) for p in EMBEDDING_PROFILES for g in GRANULARITIES]
    else:
        profile = args[0] if args else DEFAULT_PROFILE
        granularity = args[1] if len(args) > 1 else DEFAULT_GRANULARITY
        combos = [(profile, granularity)]

    for profile, granularity in combos:
        build_collection(profile, granularity)


if __name__ == "__main__":
    main()
