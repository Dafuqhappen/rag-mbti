"""Слой генерации: провайдер-агностичная обёртка над LLM.

Смысл обёртки: вызывающий код говорит "дай ответ на этот промпт" и НЕ знает,
кто его обслуживает — локальная ollama, OpenAI, DeepSeek или Anthropic.
Тогда смена провайдера — это правка конфига, а не переписывание конвейера,
и можно честно сравнить две модели на одном и том же наборе вопросов.

Ключевая мысль: у провайдеров разные форматы запроса и ответа, поэтому
каждый адаптер приводит их к одному виду. Наружу торчит одна функция complete().
"""
import json
import os
import urllib.error
import urllib.request

from config import GENERATION_PROFILES, DEFAULT_GENERATION

TIMEOUT_SEC = 300


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    """Общий HTTP-POST с JSON: у всех провайдеров транспорт одинаковый."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} вернул {exc.code}: {exc.read().decode()[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} недоступен: {exc.reason}") from exc


def _complete_ollama(system: str, user: str, config: dict, json_mode: bool) -> str:
    """Адаптер локальной ollama (/api/chat)."""
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": config.get("temperature", 0.2)},
    }
    if json_mode:
        # Ограниченное декодирование: невалидный JSON просто не может быть
        # сгенерирован. Без этого модель регулярно оборачивает ответ в
        # markdown-блок или добавляет пояснения вокруг.
        payload["format"] = "json"
    data = _post_json("http://localhost:11434/api/chat", payload)
    return data["message"]["content"].strip()


def _complete_openai_compatible(system: str, user: str, config: dict,
                                json_mode: bool) -> str:
    """Адаптер для любого OpenAI-совместимого API (OpenAI, DeepSeek, together...).

    Формат /v1/chat/completions стал де-факто стандартом, поэтому один адаптер
    обслуживает нескольких провайдеров — меняется только base_url и модель.
    """
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Не задан ключ: переменная окружения {config['api_key_env']} пуста"
        )
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.get("temperature", 0.2),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post_json(
        f"{config['base_url']}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return data["choices"][0]["message"]["content"].strip()


ADAPTERS = {
    "ollama": _complete_ollama,
    "openai-compatible": _complete_openai_compatible,
}


def complete(system: str, user: str, profile: str = DEFAULT_GENERATION,
             json_mode: bool = False) -> str:
    """Единая точка входа в генерацию. Профиль решает, кто отвечает.

    json_mode включает structured output: у каждого провайдера он называется
    по-своему (ollama — format, OpenAI — response_format), и адаптер прячет
    эту разницу. Именно ради таких расхождений обёртка и существует.
    """
    config = GENERATION_PROFILES[profile]
    return ADAPTERS[config["provider"]](system, user, config, json_mode)
