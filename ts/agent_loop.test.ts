/**
 * Тесты чистых функций цикла. Запуск (из папки ts/):  node --test
 * Модель и серверы не нужны: проверяем механику, а не поведение LLM.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { dropNulls, failureKey } from "./agent_loop.ts";

test("null значит «не задано»: поле убирается, остальные остаются", () => {
  assert.deepEqual(dropNulls({ cursor: null, only_completed: true, size: 0, q: "" }),
    { only_completed: true, size: 0, q: "" });
});

test("ложные значения — не null: 0, false и пустая строка сохраняются", () => {
  assert.deepEqual(dropNulls({ a: 0, b: false, c: "" }), { a: 0, b: false, c: "" });
});

test("повтор того же вызова с той же ошибкой даёт тот же отпечаток", () => {
  const args = { action: "recommend", input: "image-to-video" };
  assert.equal(failureKey("models_explore", args, "expected text|image"),
    failureKey("models_explore", { ...args }, "expected text|image"));
});

test("другие аргументы или другая ошибка — уже прогресс, отпечаток другой", () => {
  const base = failureKey("models_explore", { input: "image-to-video" }, "e1");
  assert.notEqual(base, failureKey("models_explore", { input: "image" }, "e1"));
  assert.notEqual(base, failureKey("models_explore", { input: "image-to-video" }, "e2"));
});
