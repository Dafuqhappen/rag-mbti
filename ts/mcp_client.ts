/**
 * MCP-клиент на TypeScript к нашему Python-серверу rag-mbti.
 *
 * Тот же разговор, что в src/mcp_raw_client.py, только через официальный SDK:
 * рукопожатие, список инструментов, вызовы, два вида ошибок.
 *
 * Запуск (из папки ts/):  node mcp_client.ts
 * Проверка типов:          npx tsc
 */
import { Client } from "@modelcontextprotocol/client";
import type { CallToolResult } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import path from "node:path";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

// Форма ответов сервера — зеркало TypedDict из src/mcp_server.py.
interface Hit {
  id: string;
  name: string;
  matched: string;
  fragment: string;
}

interface SearchResult {
  mode: string;
  degraded: boolean;
  note: string;
  hits: Hit[];
}

interface TypeCard {
  id: string;
  name: string;
  aliases: string[];
  quadra: string;
  dichotomies: { extravert: boolean; logic: boolean; intuitive: boolean; rational: boolean };
  description: string;
  functions: Record<string, string>;
}

// Интерфейс — обещание, а не проверка: во время работы типов уже нет, и TS
// поверит любому JSON. Поэтому на границе с чужим процессом проверяем руками.
// `value is SearchResult` — type guard: после true компилятор знает тип.
function isSearchResult(value: unknown): value is SearchResult {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.degraded === "boolean" && Array.isArray(v.hits);
}

function isTypeCard(value: unknown): value is TypeCard {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === "string" && typeof v.quadra === "string";
}

/** Текстовая часть результата: content — список блоков разных типов. */
function textOf(result: CallToolResult): string {
  return result.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

async function main(): Promise<void> {
  const transport = new StdioClientTransport({
    command: ".venv/bin/python",
    args: ["src/mcp_server.py"],
    cwd: PROJECT_ROOT,
    stderr: "ignore", // логи сервера идут в stderr и не нужны в выводе
  });
  const client = new Client({ name: "rag-mbti-ts-client", version: "0.1.0" });

  await client.connect(transport); // рукопожатие initialize — внутри
  try {
    const server = client.getServerVersion();
    console.log(`Подключён: ${server?.name} ${server?.version}`);

    const { tools } = await client.listTools();
    console.log(`Инструменты: ${tools.map((t) => t.name).join(", ")}\n`);

    // 1. Успешный поиск: structuredContent проверяем и только потом используем.
    const search = await client.callTool({
      name: "search_types",
      arguments: { query: "болевая этика отношений", k: 3 },
    });
    if (search.isError || !isSearchResult(search.structuredContent)) {
      throw new Error(`search_types: неожиданный ответ — ${textOf(search)}`);
    }
    const found = search.structuredContent; // здесь тип уже SearchResult
    console.log(`search_types (${found.mode}, degraded=${found.degraded}):`);
    for (const hit of found.hits) {
      console.log(`  ${hit.name} — совпало: ${hit.matched}`);
    }

    // 2. Два вызова одновременно: Promise.all — это asyncio.gather из Python.
    const [card, unknown] = await Promise.all([
      client.callTool({ name: "get_type", arguments: { name: "Дон Кихот" } }),
      client.callTool({ name: "get_type", arguments: { name: "Илон Маск" } }),
    ]);
    if (isTypeCard(card.structuredContent)) {
      const c = card.structuredContent;
      console.log(`\nget_type("Дон Кихот"): ${c.name}, квадра ${c.quadra}, псевдонимы: ${c.aliases.join(", ")}`);
    }

    // 3. Ошибка инструмента (ToolError на сервере) — НЕ исключение, а обычный
    //    результат с isError: модель агента должна его прочитать и учесть.
    console.log(`\nget_type("Илон Маск"): isError=${unknown.isError}`);
    console.log(`  ${textOf(unknown)}`);

    // 4. Несуществующий инструмент. По смыслу это протокольная ошибка, но
    //    Python SDK отдаёт её как результат с isError — поэтому код клиента
    //    обязан обрабатывать ОБА пути: исключение и isError.
    try {
      const ghost = await client.callTool({ name: "no_such_tool", arguments: {} });
      console.log(`\nno_such_tool: исключения нет, isError=${ghost.isError}: ${textOf(ghost)}`);
    } catch (err) {
      console.log(`\nno_such_tool: исключение ${err instanceof Error ? err.message : String(err)}`);
    }
  } finally {
    await client.close(); // закрывает stdin сервера, процесс Python завершается
  }
}

await main();
