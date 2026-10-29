"""Инварианты деградации search_types.

Тесты на КЛАСС поведения, а не на конкретный запрос. Сеть не нужна:
недоступность ollama имитируется подменой функции эмбеддингов. Нужна только
собранная коллекция (src/ingest.py).
"""
import pytest

import embeddings
import mcp_server
from retrieve import search

QUERY = "кто создаёт уют и избегает конфликтов"


@pytest.fixture(autouse=True)
def reset_breaker():
    mcp_server._semantic_down_until = 0.0
    yield
    mcp_server._semantic_down_until = 0.0


@pytest.fixture
def embeddings_down(monkeypatch):
    calls = {"n": 0}

    def unavailable(*args, **kwargs):
        calls["n"] += 1
        raise embeddings.EmbeddingUnavailable("ollama лежит (имитация)")

    monkeypatch.setattr(embeddings, "_embed_ollama", unavailable)
    return calls


def test_degrades_to_lexical_and_says_so(embeddings_down):
    """Недоступные эмбеддинги -> ответ есть, и он ЯВНО помечен как упрощённый."""
    result = mcp_server.search_types(QUERY, k=3)
    assert result["degraded"] is True
    assert result["mode"] == "lexical"
    assert result["note"], "модели нужно объяснение, иначе слабая выдача выглядит полной"
    assert result["hits"], "деградация обязана отвечать, а не отдавать пустоту"


def test_breaker_stops_retrying_during_cooldown(embeddings_down):
    """После сбоя повторные запросы не ходят к ollama, пока не истечёт пауза."""
    for _ in range(5):
        mcp_server.search_types(QUERY, k=2)
    assert embeddings_down["n"] == 1


def test_core_search_does_not_degrade(embeddings_down):
    """Ядро падает громко: иначе eval молча измерит BM25 под видом гибрида."""
    with pytest.raises(embeddings.EmbeddingUnavailable):
        search(QUERY, k=2)


def test_unrelated_errors_are_not_swallowed(monkeypatch):
    """Запасной путь только для недоступности эмбеддингов, не для наших багов."""
    def bug(*args, **kwargs):
        raise ValueError("ошибка в нашем коде")

    monkeypatch.setattr(embeddings, "_embed_ollama", bug)
    with pytest.raises(ValueError):
        mcp_server.search_types(QUERY, k=2)
    assert mcp_server._semantic_down_until == 0.0, "предохранитель не должен срабатывать на баг"


def test_search_uses_short_query_timeout(monkeypatch):
    """Поиск обязан ждать эмбеддинг недолго, иначе зависший хост держит каждый
    запрос по две минуты. Длинный таймаут — только для загрузки корпуса."""
    seen = {}

    def capture(texts, model, timeout):
        seen["timeout"] = timeout
        return [[0.0] * 1024 for _ in texts]

    monkeypatch.setattr(embeddings, "_embed_ollama", capture)
    search(QUERY, k=2)
    assert seen["timeout"] == embeddings.QUERY_TIMEOUT_SEC
    assert embeddings.QUERY_TIMEOUT_SEC < embeddings.BATCH_TIMEOUT_SEC
