/**
 * Агент-справочник по соционике: цикл из agent_loop.ts + наш MCP-сервер rag-mbti.
 *
 * Перенос src/agent.py один к одному по смыслу; сам цикл вынесен в
 * agent_loop.ts и не зависит от конкретного сервера.
 *
 * Запуск (из папки ts/):  node agent.ts                      # контрольные вопросы
 *                         node agent.ts "кто такой Цезарь?"  # один вопрос
 *                         node agent.ts --infra-demo         # сбой сервера посреди работы
 */
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import path from "node:path";
import { mcpToolsToLlm, runAgent, show } from "./agent_loop.ts";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

const SYSTEM = `Ты — справочник по соционике и отвечаешь с помощью инструментов.

Правила:
1. Факты о типах, функциях и квадрах НЕ бери из памяти: она ненадёжна и путает
   типы. Получай их только инструментами.
2. Даже если пользователь просит ответить коротко или одной фразой — сначала
   собери факты инструментами, а потом сформулируй коротко. Форма ответа не
   отменяет сбор фактов.
3. Если инструмент вернул ошибку — прочитай её: там написано, что делать дальше.
4. Если данных нет — так и скажи. Не придумывай.
Отвечай по-русски.`;

const CONTROL_QUESTIONS = [
  "Кто такой Цезарь в соционике? Ответь одной фразой.",
  "Какой тип: интроверт, логик, сенсорик, иррационал?",
  "Сравни ENTP с тем, кто болезненно переносит силовое давление.",
  "Какой соционический тип у Илона Маска?",
  "Расскажи про ISTK.",
  "Кто создаёт уют и избегает конфликтов?",
];

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const infraDemo = argv.includes("--infra-demo");
  const rest = argv.filter((a) => a !== "--infra-demo").join(" ");
  const questions = infraDemo ? ["Кто такой Цезарь в соционике?"] : rest ? [rest] : CONTROL_QUESTIONS;

  const transport = new StdioClientTransport({
    command: ".venv/bin/python",
    args: ["src/mcp_server.py"],
    cwd: PROJECT_ROOT,
    stderr: "ignore",
  });
  const client = new Client({ name: "rag-mbti-ts-agent", version: "0.1.0" });
  await client.connect(transport);
  try {
    const { tools } = await client.listTools();
    const llmTools = mcpToolsToLlm(tools);
    // Демонстрация сбоя: сервер «падает» раньше, чем модель успеет вызвать инструмент.
    if (infraDemo) await client.close();
    // Вопросы — по очереди, не через Promise.all: у локальной ollama один слот
    // (OLLAMA_NUM_PARALLEL=1), параллельный клиент лишь выстроит очередь у сервера.
    for (const q of questions) show(q, await runAgent(q, client, llmTools, { system: SYSTEM }));
  } finally {
    await client.close();
  }
}

await main();
