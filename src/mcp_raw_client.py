"""MCP «на голом протоколе»: клиент без SDK, чтобы увидеть, что под капотом.

Запуск:  .venv/bin/python src/mcp_raw_client.py

Делает ровно то, что делает Claude Code при подключении сервера: запускает его
дочерним процессом и обменивается строками JSON-RPC 2.0 через stdin/stdout.
Одно сообщение = одна строка JSON. Ничего больше в протоколе stdio нет.
"""
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PROTOCOL_VERSION = "2026-07-28"


class RawClient:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose  # печатать ли каждое сообщение протокола
        # stderr сервера (логи) отбрасываем: протокол живёт только в stdout
        self.proc = subprocess.Popen(
            [sys.executable, str(SRC / "mcp_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", cwd=SRC,
        )
        self.next_id = 1

    def _send(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False)
        if self.verbose:
            print(f"\n→ {_short(line)}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        """Запрос: есть id, сервер обязан ответить сообщением с тем же id."""
        msg_id, self.next_id = self.next_id, self.next_id + 1
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                    **({"params": params} if params is not None else {})})
        while True:
            reply = json.loads(self.proc.stdout.readline())
            if self.verbose:
                print(f"← {_short(json.dumps(reply, ensure_ascii=False))}")
            if reply.get("id") == msg_id:
                return reply

    def notify(self, method: str) -> None:
        """Уведомление: id нет, ответа не будет и ждать его нельзя."""
        self._send({"jsonrpc": "2.0", "method": method})

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def _short(text: str, limit: int = 420) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def main() -> None:
    c = RawClient()

    print("=" * 30, "1. рукопожатие", "=" * 30)
    c.request("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "raw-client", "version": "0.1"},
    })
    c.notify("notifications/initialized")

    print("\n" + "=" * 30, "2. что умеет сервер", "=" * 30)
    tools = c.request("tools/list")["result"]["tools"]
    print("\nинструменты:", [t["name"] for t in tools])
    print("схема аргументов filter_types (её и видит модель):")
    schema = next(t for t in tools if t["name"] == "filter_types")["inputSchema"]
    print(json.dumps(schema["properties"], ensure_ascii=False))

    print("\n" + "=" * 30, "3. вызовы инструментов", "=" * 30)
    c.request("tools/call", {"name": "filter_types", "arguments": {
        "energy": "extravert", "judgment": "logic", "perception": "intuitive",
        "lifestyle": "irrational"}})
    c.request("tools/call", {"name": "search_types", "arguments": {
        "query": "у кого болевая этика отношений", "k": 2}})

    print("\n" + "=" * 30, "4. две разные ошибки", "=" * 30)
    print("(а) ToolError — ожидаемая, сервер сам её бросил:")
    c.request("tools/call", {"name": "filter_types", "arguments": {}})
    print("\n(б) несуществующий метод — ошибка уровня протокола:")
    c.request("no/such_method")

    print("\n" + "=" * 30, "5. ресурсы и промпты", "=" * 30)
    c.request("resources/templates/list")
    c.request("prompts/list")

    c.close()


if __name__ == "__main__":
    main()
