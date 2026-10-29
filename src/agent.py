"""Агентный цикл своими руками: LLM + инструменты нашего MCP-сервера.

Запуск:  .venv/bin/python src/agent.py                       # шесть контрольных вопросов
         .venv/bin/python src/agent.py "кто такой Цезарь?"   # один вопрос

Агент — это не особая модель, а ЦИКЛ вокруг обычной:

    сообщения = [system, вопрос]
    повторять:
        ответ = LLM(сообщения, инструменты)
        есть просьбы вызвать инструменты -> выполнить, результаты дописать, повторить
        нет просьб                       -> это финальный ответ
        слишком много шагов              -> остановиться

Модель ничего не выполняет сама: она только просит. Выполняет оркестратор —
этот файл. LangGraph, CrewAI и подобные — это более удобные версии того же цикла.

Агент здесь — MCP-клиент нашего же сервера. Список инструментов берётся из
tools/list, а inputSchema из MCP без изменений становится описанием функции для
модели: это тот же формат JSON Schema. Поэтому MCP так легко встраивается в
агентов.
"""
import json
import sys
import urllib.request

from mcp_raw_client import PROTOCOL_VERSION, RawClient

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
MAX_STEPS = 6   # предохранитель от бесконечного цикла: модель может звать инструменты вечно
MAX_REPAIRS = 2  # сколько раз даём модели переделать битый вызов, прежде чем сдаться

# Признаки того, что модель хотела вызвать инструмент, но написала вызов текстом.
LEAK_MARKERS = ("<tool_call>", "</tool_call>", '"arguments"')

REPAIR_MESSAGE = (
    "Твой вызов инструмента не распознан: он пришёл текстом внутри ответа. "
    "Вызови инструмент через механизм вызова инструментов, без текста вокруг. "
    "Если инструмент не нужен — просто ответь пользователю."
)
GIVE_UP_ANSWER = "Не удалось обработать запрос: модель не смогла корректно вызвать инструмент."

SYSTEM = """Ты — справочник по соционике и отвечаешь с помощью инструментов.

Правила:
1. Факты о типах, функциях и квадрах НЕ бери из памяти: она ненадёжна и путает
   типы. Получай их только инструментами.
2. Даже если пользователь просит ответить коротко или одной фразой — сначала
   собери факты инструментами, а потом сформулируй коротко. Форма ответа не
   отменяет сбор фактов.
3. Если инструмент вернул ошибку — прочитай её: там написано, что делать дальше.
4. Если данных нет — так и скажи. Не придумывай.
Отвечай по-русски."""


def mcp_tools_to_llm(tools: list[dict]) -> list[dict]:
    """Инструменты MCP -> формат function calling. inputSchema переносится как есть."""
    return [
        {"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["inputSchema"],
        }}
        for t in tools
    ]


def llm(messages: list[dict], tools: list[dict]) -> dict:
    """Один шаг модели: она отвечает текстом или просит вызвать инструменты."""
    body = {"model": MODEL, "messages": messages, "tools": tools,
            "stream": False, "options": {"temperature": 0}}
    request = urllib.request.Request(
        OLLAMA_CHAT_URL, json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]


def call_tool(client: RawClient, name: str, arguments: dict) -> tuple[str, bool]:
    """Выполняет инструмент через MCP. Возвращает (текст для модели, была ли ошибка).

    Ошибку инструмента отдаём модели как обычный результат, а не роняем цикл:
    текст ошибки — подсказка, по которой модель может исправиться (другой
    инструмент, другие аргументы). Так агент чинит себя сам.
    """
    reply = client.request("tools/call", {"name": name, "arguments": arguments})
    if "error" in reply:  # ошибка уровня протокола, например несуществующий инструмент
        return f"Ошибка: {reply['error'].get('message')}", True
    result = reply["result"]
    text = "\n".join(c.get("text", "") for c in result.get("content", []))
    if result.get("isError"):
        return text, True
    structured = result.get("structuredContent")
    return (json.dumps(structured, ensure_ascii=False) if structured is not None else text), False


def looks_like_leaked_call(text: str) -> bool:
    """Модель хотела вызвать инструмент, но написала вызов текстом."""
    return any(marker in text for marker in LEAK_MARKERS)


def run_agent(question: str, client: RawClient, tools: list[dict],
              max_steps: int = MAX_STEPS, max_repairs: int = MAX_REPAIRS) -> dict:
    """Сам агентный цикл. Возвращает ответ и трассу: какие инструменты звались."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    trace = []
    repairs = 0

    for step in range(1, max_steps + 1):
        message = llm(messages, tools)
        messages.append(message)
        calls = message.get("tool_calls") or []

        if not calls:
            text = (message.get("content") or "").strip()
            if not looks_like_leaked_call(text):  # обычный текст -> финальный ответ
                return {"answer": text, "trace": trace, "steps": step}

            # Вызов пришёл текстом. Разбирать и исполнять его НЕЛЬЗЯ: текст —
            # не команда. Если оркестратор исполняет вызовы из свободного текста,
            # любой, кто подложил строку в контекст (документ, результат другого
            # инструмента), получает возможность «напечатать» вызов — это prompt
            # injection. Исполняем только структурированные вызовы; модели даём
            # ограниченное число попыток переделать, потом честно сдаёмся.
            trace.append({"step": step, "tool": "(битый вызов)", "args": {},
                          "error": True, "result": text[:200]})
            if repairs >= max_repairs:
                return {"answer": GIVE_UP_ANSWER, "trace": trace, "steps": step,
                        "stopped": "malformed_tool_call"}
            repairs += 1
            messages.append({"role": "user", "content": REPAIR_MESSAGE})
            continue

        for call in calls:
            fn = call["function"]
            args = fn.get("arguments") or {}
            # ollama отдаёт аргументы словарём, OpenAI-совместимые API — строкой JSON
            if isinstance(args, str):
                args = json.loads(args)
            result, is_error = call_tool(client, fn["name"], args)
            trace.append({"step": step, "tool": fn["name"], "args": args,
                          "error": is_error, "result": result})
            messages.append({"role": "tool", "tool_name": fn["name"], "content": result})

    return {"answer": None, "trace": trace, "steps": max_steps, "stopped": "max_steps"}


def connect() -> tuple[RawClient, list[dict]]:
    client = RawClient(verbose=False)
    client.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                  "clientInfo": {"name": "agent", "version": "0.1"}})
    client.notify("notifications/initialized")
    tools = client.request("tools/list")["result"]["tools"]
    return client, mcp_tools_to_llm(tools)


def show(question: str, result: dict) -> None:
    print("=" * 78)
    print(f"ВОПРОС: {question}")
    if not result["trace"]:
        print("  (инструменты не вызывались)")
    for t in result["trace"]:
        mark = "✗" if t["error"] else "✓"
        if t["tool"] == "(битый вызов)":
            print(f"  шаг {t['step']} → битый вызов текстом, отправлен на исправление ✗")
            continue
        args = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
        print(f"  шаг {t['step']} → {t['tool']}({args}) {mark}  {t['result'][:110]}")
    answer = result["answer"] if result["answer"] is not None else f"[остановлен: {result['stopped']}]"
    print(f"ОТВЕТ ({result['steps']} шаг.): {answer}")


CONTROL_QUESTIONS = [
    "Кто такой Цезарь в соционике? Ответь одной фразой.",
    "Какой тип: интроверт, логик, сенсорик, иррационал?",
    "Сравни ENTP с тем, кто болезненно переносит силовое давление.",
    "Какой соционический тип у Илона Маска?",
    "Расскажи про ISTK.",
    "Кто создаёт уют и избегает конфликтов?",
]

if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else CONTROL_QUESTIONS
    mcp, llm_tools = connect()
    try:
        for q in questions:
            show(q, run_agent(q, mcp, llm_tools))
    finally:
        mcp.close()
