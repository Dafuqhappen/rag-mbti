"""Слой эмбеддингов: превращение текста в векторы.

Вынесен отдельно намеренно. Это провайдер-агностичная обёртка: код выше
не знает, кто считает векторы — Chroma своей встроенной моделью, локальная
ollama или облачный API. Чтобы сравнить две модели, достаточно сменить
профиль, а не переписывать конвейер.
"""
import json
import os
import urllib.error
import urllib.request

from config import EMBEDDING_PROFILES

# Адрес из окружения: так сервис можно перенести на другой хост без правки кода,
# а в тестах — сымитировать недоступность, указав мёртвый порт.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_URL}/api/embed"

# Таймауты подобраны замером (bge-m3, Mac, модель на GPU):
#   один запрос   — медиана 106 мс, худший 132 мс; холодный старт 1.1 с
#   батч 144 чанка — 3.4 с
# Один общий таймаут не подходит: 3 с обрывали бы загрузку корпуса, а 120 с
# заставляли бы каждый поиск ждать две минуты, если хост завис.
QUERY_TIMEOUT_SEC = 3     # ~25x тёплого худшего случая, ~3x холодного старта
BATCH_TIMEOUT_SEC = 120   # загрузка корпуса; запас ~35x на рост корпуса


class EmbeddingUnavailable(RuntimeError):
    """Провайдер эмбеддингов недоступен: сервис лежит, модели нет, таймаут.

    Отдельный класс нужен, чтобы вызывающий код мог поймать ИМЕННО эту ситуацию
    и деградировать. Ловить ради этого весь RuntimeError нельзя: под запасной
    путь молча уйдёт любая ошибка в нашем собственном коде.
    """


def _embed_ollama(texts: list[str], model: str, timeout: float) -> list[list[float]]:
    """Считает эмбеддинги батчем через локальный ollama."""
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_EMBED_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["embeddings"]
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EmbeddingUnavailable(
            f"ollama недоступна на {OLLAMA_EMBED_URL}: {exc}. Запусти `ollama serve`."
        ) from exc


def embed(texts: list[str], profile: str,
          timeout: float = QUERY_TIMEOUT_SEC) -> list[list[float]] | None:
    """Векторы для списка текстов.

    timeout по умолчанию рассчитан на поисковый запрос. Загрузка корпуса
    батчем обязана передать BATCH_TIMEOUT_SEC: решает вызывающий, потому что
    только он знает, что именно он отправляет.

    Возвращает None для профиля chroma-default: там векторы считает сама Chroma
    внутри add()/query(), и подставлять их снаружи не нужно.
    """
    config = EMBEDDING_PROFILES[profile]
    if config["provider"] == "ollama":
        return _embed_ollama(texts, config["model"], timeout)
    return None
