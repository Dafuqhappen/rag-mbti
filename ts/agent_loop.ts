/**
 * Агентный цикл: ollama (qwen) + инструменты ЛЮБОГО MCP-сервера.
 *
 *     сообщения = [system, вопрос]
 *     повторять:
 *         ответ = LLM(сообщения, инструменты)
 *         есть вызовы инструментов -> выполнить, результаты дописать, повторить
 *         нет вызовов              -> это финальный ответ
 *         слишком много шагов      -> остановиться
 *
 * Цикл не знает, к какому серверу подключён: другой MCP-сервер (stdio или
 * HTTP) подключается своим промптом и белым списком, код цикла тот же.
 *
 * Разбор ошибок по вопросу «кто может исправить»:
 *   - isError, ProtocolError, инструмент вне белого списка, битые аргументы —
 *     ошибка модели: текст уходит ей, цикл продолжается;
 *   - всё остальное (сервер упал, обрыв, таймаут) — сбой инфраструктуры: модель
 *     его не исправит, агент останавливается и честно отказывает.
 */
import { Client, ProtocolError, SdkError } from "@modelcontextprotocol/client";
import type { CallToolResult, Tool } from "@modelcontextprotocol/client";

const OLLAMA_CHAT_URL = "http://localhost:11434/api/chat";
const MODEL = "qwen2.5:7b";
const LLM_TIMEOUT_MS = 300_000;

const MAX_STEPS = 6; // предохранитель от бесконечного цикла
const MAX_REPAIRS = 2; // сколько раз даём модели переделать вызов, написанный текстом

// Признаки того, что модель хотела вызвать инструмент, но написала вызов текстом.
const LEAK_MARKERS = ["<tool_call>", "</tool_call>", '"arguments"'];

const REPAIR_MESSAGE =
  "Твой вызов инструмента не распознан: он пришёл текстом внутри ответа. " +
  "Вызови инструмент через механизм вызова инструментов, без текста вокруг. " +
  "Если инструмент не нужен — просто ответь пользователю.";
const GIVE_UP_ANSWER = "Не удалось обработать запрос: модель не смогла корректно вызвать инструмент.";
const INFRA_ANSWER = "Сервис сейчас недоступен. Попробуйте позже.";
const NO_PROGRESS_ANSWER = "Не удалось получить данные: инструмент раз за разом возвращает одну и ту же ошибку.";

// --- Формат сообщений ollama /api/chat ---------------------------------------

interface ToolCall {
  function: { name: string; arguments: Record<string, unknown> | string };
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: ToolCall[];
  tool_name?: string;
}

export interface LlmTool {
  type: "function";
  function: { name: string; description: string; parameters: unknown };
}

export interface TraceEntry {
  step: number;
  tool: string;
  args: Record<string, unknown>;
  error: boolean;
  result: string;
}

// Discriminated union: поле `kind` говорит, какие ещё поля есть. После проверки
// `result.kind === "answer"` компилятор знает, что `answer` — строка, а `reason`
// у этого варианта нет. В Python то же самое держалось на соглашении
// («answer is None, тогда смотри stopped»), и нарушить его можно было молча.
export type AgentResult =
  | { kind: "answer"; answer: string; trace: TraceEntry[]; steps: number; promptTokens: number }
  | {
      kind: "stopped";
      reason: "max_steps" | "malformed_tool_call" | "infrastructure" | "no_progress";
      answer: string | null;
      trace: TraceEntry[];
      steps: number;
      promptTokens: number;
    };

export interface AgentOptions {
  system: string;
  // Белый список исполнителя. Модели и так показаны только разрешённые
  // инструменты, но имя она может написать любое — или его подсунет текст из
  // результата (prompt injection). Исполнитель проверяет сам, не веря модели.
  allowedTools?: ReadonlySet<string>;
  numCtx?: number; // окно контекста ollama; без него — умолчание модели
  maxResultChars?: number; // длинные результаты обрезаются ЯВНО, с пометкой
}

// --- Граница с ollama: ответ из чужого процесса проверяем руками --------------

function isChatMessage(value: unknown): value is ChatMessage {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.role === "string" && typeof v.content === "string" &&
    (v.tool_calls === undefined || Array.isArray(v.tool_calls));
}

/** Инструменты MCP -> формат function calling. inputSchema переносится как есть. */
export function mcpToolsToLlm(tools: Tool[]): LlmTool[] {
  return tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description ?? "", parameters: t.inputSchema },
  }));
}

/** Один шаг модели: она отвечает текстом или просит вызвать инструменты. */
async function llm(
  messages: ChatMessage[],
  tools: LlmTool[],
  numCtx: number | undefined,
): Promise<{ message: ChatMessage; promptTokens: number }> {
  const options = numCtx === undefined ? { temperature: 0 } : { temperature: 0, num_ctx: numCtx };
  const response = await fetch(OLLAMA_CHAT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages, tools, stream: false, options }),
    signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`ollama: HTTP ${response.status}`);
  const body = (await response.json()) as { message?: unknown; prompt_eval_count?: unknown };
  if (!isChatMessage(body.message)) throw new Error("ollama: неожиданный формат ответа");
  const promptTokens = typeof body.prompt_eval_count === "number" ? body.prompt_eval_count : 0;
  return { message: body.message, promptTokens };
}

function textOf(result: CallToolResult): string {
  return result.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

/**
 * Выполняет инструмент. Возвращает текст для модели и флаг ошибки.
 *
 * Ошибки, которые модель может исправить, возвращаются ей обычным результатом.
 * Сбой инфраструктуры пробрасывается наверх: модель тут ни при чём, и решать,
 * что делать, должен цикл, а не она.
 */
async function callTool(
  client: Client,
  name: string,
  args: Record<string, unknown>,
): Promise<{ text: string; isError: boolean }> {
  try {
    const result = await client.callTool({ name, arguments: args });
    if (result.isError) return { text: textOf(result), isError: true };
    const structured = result.structuredContent;
    return { text: structured !== undefined ? JSON.stringify(structured) : textOf(result), isError: false };
  } catch (err) {
    // Сервер ответил JSON-RPC ошибкой (другой сервер может так отдать и
    // неизвестный инструмент): это про запрос модели — пусть она и исправляет.
    if (err instanceof ProtocolError) return { text: `Ошибка: ${err.message}`, isError: true };
    throw err; // SdkError и всё прочее — не к модели
  }
}

/** Аргументы: ollama отдаёт объект, OpenAI-совместимые API — строку JSON. */
function parseArgs(raw: ToolCall["function"]["arguments"]): Record<string, unknown> | null {
  if (typeof raw !== "string") return raw;
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/**
 * Поля со значением null убираем: для модели null значит «не задано», а схема
 * сервера часто null не принимает (`cursor: null` -> Invalid input). Меняется
 * только форма, смысл тот же — это нормализация, а не исправление за модель.
 */
export function dropNulls(args: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(args).filter(([, value]) => value !== null));
}

/** Отпечаток неудачного вызова: тот же инструмент, те же аргументы, та же ошибка. */
export function failureKey(name: string, args: Record<string, unknown>, error: string): string {
  return JSON.stringify([name, args, error]);
}

function looksLikeLeakedCall(text: string): boolean {
  return LEAK_MARKERS.some((marker) => text.includes(marker));
}

function clip(text: string, limit: number | undefined): string {
  if (limit === undefined || text.length <= limit) return text;
  return `${text.slice(0, limit)}\n[результат обрезан: показано ${limit} из ${text.length} символов]`;
}

// --- Сам цикл ----------------------------------------------------------------

export async function runAgent(
  question: string,
  client: Client,
  tools: LlmTool[],
  opts: AgentOptions,
): Promise<AgentResult> {
  const messages: ChatMessage[] = [
    { role: "system", content: opts.system },
    { role: "user", content: question },
  ];
  const trace: TraceEntry[] = [];
  let repairs = 0;
  let lastFailure: string | null = null; // для детектора «нет прогресса»
  let promptTokens = 0; // максимум за прогон: столько занял самый длинный шаг

  for (let step = 1; step <= MAX_STEPS; step++) {
    const turn = await llm(messages, tools, opts.numCtx);
    promptTokens = Math.max(promptTokens, turn.promptTokens);
    const message = turn.message;
    messages.push(message);
    const calls = message.tool_calls ?? [];

    if (calls.length === 0) {
      const text = message.content.trim();
      if (!looksLikeLeakedCall(text)) return { kind: "answer", answer: text, trace, steps: step, promptTokens };

      // Вызов пришёл текстом. Исполнять его нельзя: текст — не команда, иначе
      // любая строка в контексте (документ, чужой результат) может «напечатать»
      // вызов — это prompt injection. Даём ограниченное число попыток переделать.
      trace.push({ step, tool: "(битый вызов)", args: {}, error: true, result: text.slice(0, 200) });
      if (repairs >= MAX_REPAIRS) {
        return { kind: "stopped", reason: "malformed_tool_call", answer: GIVE_UP_ANSWER, trace, steps: step, promptTokens };
      }
      repairs++;
      messages.push({ role: "user", content: REPAIR_MESSAGE });
      continue;
    }

    for (const call of calls) {
      const name = call.function.name;
      const parsed = parseArgs(call.function.arguments);
      const args = parsed === null ? null : dropNulls(parsed);
      let outcome: { text: string; isError: boolean };
      if (opts.allowedTools && !opts.allowedTools.has(name)) {
        outcome = { text: `Ошибка: инструмент «${name}» недоступен. Доступны: ${[...opts.allowedTools].join(", ")}.`, isError: true };
      } else if (args === null) {
        outcome = { text: "Ошибка: аргументы не являются JSON-объектом. Передай аргументы объектом.", isError: true };
      } else {
        try {
          outcome = await callTool(client, name, args);
        } catch (err) {
          // Сбой инфраструктуры: модели не показываем, цикл останавливаем.
          const why = err instanceof SdkError ? `${err.code}: ${err.message}` : String(err);
          trace.push({ step, tool: name, args, error: true, result: `[инфраструктура] ${why}` });
          return { kind: "stopped", reason: "infrastructure", answer: INFRA_ANSWER, trace, steps: step, promptTokens };
        }
      }
      const text = clip(outcome.text, opts.maxResultChars);
      trace.push({ step, tool: name, args: args ?? {}, error: outcome.isError, result: text });

      // Нет прогресса: модель повторила ровно тот вызов, который только что
      // упал с той же ошибкой. При temperature 0 дальше будет то же самое —
      // систематическая ошибка повтором не лечится. Останавливаемся честно,
      // а не сжигаем оставшиеся шаги.
      const key = outcome.isError ? failureKey(name, args ?? {}, outcome.text) : null;
      if (key !== null && key === lastFailure) {
        return { kind: "stopped", reason: "no_progress", answer: NO_PROGRESS_ANSWER, trace, steps: step, promptTokens };
      }
      lastFailure = key;
      messages.push({ role: "tool", tool_name: name, content: text });
    }
  }
  return { kind: "stopped", reason: "max_steps", answer: null, trace, steps: MAX_STEPS, promptTokens };
}

export function show(question: string, result: AgentResult): void {
  console.log("=".repeat(78));
  console.log(`ВОПРОС: ${question}`);
  if (result.trace.length === 0) console.log("  (инструменты не вызывались)");
  for (const t of result.trace) {
    if (t.tool === "(битый вызов)") {
      console.log(`  шаг ${t.step} → битый вызов текстом, отправлен на исправление ✗`);
      continue;
    }
    const args = Object.entries(t.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
    console.log(`  шаг ${t.step} → ${t.tool}(${args}) ${t.error ? "✗" : "✓"}  ${t.result.slice(0, 110)}`);
  }
  const answer = result.kind === "answer" ? result.answer : `[остановлен: ${result.reason}] ${result.answer ?? ""}`;
  console.log(`ОТВЕТ (${result.steps} шаг., контекст до ${result.promptTokens} ток.): ${answer}`);
}
