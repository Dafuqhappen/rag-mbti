"""Единая точка настроек — пути и ручки качества.

Держим их здесь, а не россыпью по коду: конфигурация эксперимента (какая
модель эмбеддингов, какая гранулярность, какой вес слияния) должна меняться
в одном месте, иначе сравнивать варианты честно не получится.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"      # векторное хранилище сохраняется на диск сюда

EMBEDDING_PROFILES = {
    # Встроенная модель Chroma (all-MiniLM-L6-v2, 384 измерения) — англоязычная.
    "minilm": {"provider": "chroma-default"},
    # Многоязычная модель через локальную ollama (1024 измерения).
    "bge-m3": {"provider": "ollama", "model": "bge-m3"},
}
DEFAULT_PROFILE = "bge-m3"

# Гранулярность индексации: whole — тип целиком, chunked — описание и каждая
# функция отдельным чанком. При chunked работает small-to-big: ищем по чанкам,
# а возвращаем документ-родитель целиком.
GRANULARITIES = ("whole", "chunked")
DEFAULT_GRANULARITY = "chunked"

# Вес семантического ранкера зависит от ПАРЫ (модель, гранулярность), потому
# что оба фактора меняют его силу. Значения подобраны разверткой
# (sweep.py <профиль> <гранулярность>), берётся устойчивое плато, а не
# одиночный пик: на 22 примерах пик почти наверняка подгонка под шум.
# Показательно, что на whole семантику приходится глушить (0.3), а на chunked
# ей полагается равный вес: короткий фокусный чанк она понимает куда лучше,
# чем большой однородный документ.
SEMANTIC_WEIGHTS = {
    ("minilm", "whole"): 0.4,
    ("minilm", "chunked"): 1.0,
    ("bge-m3", "whole"): 0.3,
    ("bge-m3", "chunked"): 1.0,
}
LEXICAL_WEIGHT = 1.0               # опорный вес BM25, семантика взвешивается относительно него


def collection_name(profile: str, granularity: str) -> str:
    """Каждая комбинация живёт в своей коллекции, чтобы сравнивать их рядом."""
    return f"mbti_{profile.replace('-', '_')}_{granularity}"


def semantic_weight(profile: str, granularity: str) -> float:
    return SEMANTIC_WEIGHTS[(profile, granularity)]


TOP_K = 3                          # сколько документов возвращает ретривал


# Профили генерации. Устроены так же, как профили эмбеддингов: код выше не
# знает, кто отвечает — локальная модель или облачный API. Смена провайдера
# должна быть правкой конфига, а не переписыванием конвейера.
# temperature низкая: в справочной задаче нам нужна точность, а не фантазия.
GENERATION_PROFILES = {
    "local": {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "temperature": 0.2,
    },
    # Пример облачного провайдера. Ключ берётся из переменной окружения,
    # в код и в git не попадает никогда.
    "deepseek": {
        "provider": "openai-compatible",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.2,
    },
}
DEFAULT_GENERATION = "local"

# Какой системный промпт используем (варианты — в prompts.py).
DEFAULT_PROMPT = "strict"
