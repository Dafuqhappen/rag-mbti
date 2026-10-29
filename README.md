# rag-mbti — a RAG reference on socionics

**English** · [Русский](README.ru.md)

A learning and research project: a RAG system over a corpus of the 16 socionics
types (a Russian personality typology close to MBTI), an MCP server, and a
hand-written agent loop in Python and TypeScript. The point is not the bot
itself but **the measurement method** — for retrieval, generation and agent
behaviour. Every change is backed by a measurement, not by a feeling.

The corpus, queries and prompts are in Russian.

## Results

| Retrieval metric | Baseline | Final | Gain |
|---|---|---|---|
| Recall@3 | 0.398 | **0.932** | +134% |
| Recall@5 | 0.670 | **1.000** | +49% |
| MRR | 0.383 | **0.966** | +152% |

Measured on a golden set of 22 queries labelled in 8 categories.

### What made the difference

| Change | Cost | Recall@3 contribution |
|---|---|---|
| BM25 + hybrid fusion | free | +0.295 |
| Chunking + small-to-big | free | +0.193 |
| Switching to a multilingual embedding model | 1.2 GB + a dependency | +0.023 (noise) |
| Tuning the fusion weight | free | +0.023 (noise) |
| Auto-filter by dichotomies | +2–5 s per query | 0 at k=3, +0.045 at k=5 |

Main takeaway: **corpus structure and the choice of search algorithm gave almost
all of the improvement; a heavier embedding model was statistical noise.**

## Architecture

```
question
  │
  ├─ retrieval (retrieve.py)
  │    ├─ semantic  — vector search, embeddings via ollama or Chroma
  │    ├─ lexical   — BM25 with a truncation stemmer (lexical.py)
  │    └─ hybrid    — weighted Reciprocal Rank Fusion
  │         └─ small-to-big: search over chunks, return the whole type
  │
  └─ generation (rag.py -> generate.py)
       └─ provider-agnostic wrapper: ollama / any OpenAI-compatible API

interfaces:
  bot.py         — Telegram, long polling
  mcp_server.py  — MCP server: the reference as tools for any agent

agents (clients of the MCP server):
  agent.py           — agent loop in Python, ollama + qwen2.5-7b
  ts/agent_loop.ts   — the same loop in TypeScript, server-agnostic
```

### Key decisions

- **Hybrid search.** Vector search is bad at rare exact tokens (names, type
  codes); BM25 is the opposite. RRF fuses ranks, not scores (scores live on
  different scales): `weight / (60 + rank)`.
- **The fusion weight depends on the configuration.** On whole documents the
  semantic ranker has to be damped (0.3); on chunks it deserves equal weight
  (1.0) — it understands short focused text better. Weights were chosen by a
  sweep (`sweep.py`).
- **Small-to-big.** Small chunks are good for search, large ones for answering.
  We index chunks and return the parent document.
- **Metadata filtering.** Structural queries ("intuitive + logical + extravert")
  cannot be solved by vector search at all — they need an exact metadata filter.
- **Query understanding with a gatekeeper.** Structural queries are parsed by an
  LLM (structured output + validation + fail-open), but two deterministic checks
  run first: is a specific type named, and is a Model A slot named. Without
  them the LLM built filters where they hurt and dropped quality by 0.137.
  Don't ask the model to decide what a dictionary lookup decides.
- **Blocking call outside the event loop.** A RAG answer takes ~10 s
  synchronously. In an async bot that would freeze every user, so it runs in a
  thread via `asyncio.to_thread`.
- **Corpus validation.** Model A functions are derived from the leading and
  creative function by a strict rule, and a script checks the invariant:
  LLM-generated data is a draft, not data.

## Metrics

**Retrieval:** Recall@k (did we find everything) and MRR (how high the right
answer ranks). The pair is deliberate: Hit Rate loses information, and each
metric alone hides a different kind of failure.

**Generation:** groundedness (no invented entities — a programmatic check
instead of an LLM judge, since the domain is closed), citation rate, and refusal
rate on questions the corpus cannot answer.

## MCP server

The reference is available to any MCP client (Claude Code, Claude Desktop,
Cursor). The connection is already described in `.mcp.json`: open the repo in
Claude Code and approve the server.

| Primitive | What | Who decides to use it |
|---|---|---|
| tool `search_types` | hybrid search for a free-text question | model |
| tool `filter_types` | exact selection by dichotomies (word-valued parameters, see below) | model |
| tool `get_type` | a type card by any name, declension or MBTI letters | model |
| resource `mbti://types/{id}` | full text of a type | application |
| prompt `compare_types` | compare two types along Model A | user |

Decisions:

- **The server returns data, not reasoning.** There is no answer generation:
  the client is a strong model.
- **The error text is a hint for the model.** On an ambiguous name `get_type`
  lists the candidates instead of guessing; on a typo in an MBTI code it
  suggests the closest codes; on a person's name it explicitly forbids typing
  that person.
- **Graceful degradation.** Without ollama `search_types` answers via BM25 and
  marks the answer `degraded`. The price is measured: recall@3 unchanged, MRR
  0.955 → 0.871. A circuit breaker skips ollama for 30 s after a failure. The
  `retrieve` core still fails loudly, so evaluations never silently run on the
  fallback.
- **Separate timeouts** from measurements: a query — 3 s (median 106 ms, cold
  start 1.1 s), corpus loading — 120 s (a batch of 144 chunks takes 3.4 s).

`src/mcp_raw_client.py` talks to the server over raw JSON-RPC without the SDK —
it shows the protocol line by line. `ts/mcp_client.ts` does the same through the
official TypeScript SDK.

## Generation

A provider-agnostic wrapper: ollama locally or any OpenAI-compatible API
(switching is a config change). The prompt lives separately in two variants so
they can be compared by measurement.

| Metric | strict | loose (naive prompt) |
|---|---|---|
| groundedness | 1.00 | 1.00 |
| citation | 0.75 | 0.00 |
| refusal on questions without data | 0.83 | **0.00** |
| false refusals | 0.00 | 0.00 |
| sec/answer | 11.9 | 19.6 |

The naive prompt made up an answer to **every** question the corpus has no data
for: it invented Elon Musk's socionics type, a "1–2% of the population" figure
citing "modern research", and a non-existent founder of socionics.

Tellingly, **groundedness did not notice** (1.00 for both): it checks entities,
and the entities came from the context — it was the claim that was invented.
Only the refusal metric caught it. Hence: a metric set has to cover different
failure modes, because every single metric is blind to something.

## Agent loop

An agent is not a special model but a loop around an ordinary one: the model
either answers or asks for a tool call; the program executes it, appends the
result and asks again — until there is an answer or the step limit runs out.
`src/agent.py` and `ts/agent_loop.ts` are the same loop with no framework, as a
client of our MCP server.

Decisions:

- **Text is not a command.** A tool call printed as text inside the answer is
  never parsed or executed — otherwise any string in the context could "print"
  a call (prompt injection). The model gets a limited number of attempts to
  redo the call, then an honest refusal.
- **Errors are routed by "who can fix it".** Tool errors, the wrong tool or bad
  arguments go back to the model as text, and it fixes them itself.
  Infrastructure failures (server down, broken connection) are not shown to the
  model: the loop stops with an honest refusal. This is an allowlist: only the
  listed errors go to the model; everything else stops the loop. Verified
  (`node agent.ts --infra-demo`): if every error is forwarded, then with the
  server down the model answers from memory within 24 s and gets it wrong,
  while monitoring sees a "successful" answer.
- **Executor-side tool allowlist** — a check in code, not in the prompt: a tool
  name can be smuggled in by text from a tool result.
- **No progress — stop.** The same call with the same error twice in a row at
  temperature 0 will keep repeating: the loop stops (6 steps → 2).
- **`null` in arguments means "not set"**: small models write optional fields
  that way, and server schemas often reject `null`.

On qwen2.5-7b: 3 of 6 control questions, with failures of three classes: wrong
argument, incomplete plan, call printed as text. After porting the loop to
TypeScript the classes repeated one for one: the failures come from the model
and what it sees, not from the loop code.

## Tool design: a measurement

For "introvert, logical, **sensing**, irrational" the model passed
`is_intuitive=true`: a boolean requires a negation ("sensing" = NOT intuitive).
`src/eval_tool_design.py` measures only the model's first call — the shape of
the `filter_types` parameters with an identical description — on 16 questions
across all 6 tool orders:

| Variant | exact | inverted features |
|---|---|---|
| boolean (`is_intuitive: true/false`) | 7–13 of 16, mean 10.5 | 12 |
| words (`perception: "intuitive" \| "sensing"`) | 13–15 of 16, mean 14.2 | 0 |

- **A hidden variable.** The first run showed 10 vs 15, but the boolean tool was
  second in the list and the enum one last. Moving the boolean tool alone shifted
  the result by 3 questions — hence the measurement across all orders.
- An enum with Russian values ("сенсорик") was no better than English: the model
  copied the negation from the question too ("not logical"). What helped was
  removing the negation.
- An out-of-list value is rejected by the schema itself (`Literal`) with the
  list of allowed values, not silently widened into a broader filter
  (`tests/test_filter_types.py`).

**A better component is not a better system.** After the switch to enum, a
full agent run fixed the target question but exposed another defect: search
always returns k types, so the agent assigns types to real people
(`src/eval_people.py`: 6 of 7). A relevance threshold was rejected by a
counter-check before writing code: distances for normal and off-topic queries
overlap, and a good semantic query ("a born orator") has BM25 = 0. Changing the
tool description did not help and broke a neighbouring question — reverted.
The defect is open.

## asyncio: measurements

| Demo | What it shows | Result |
|---|---|---|
| `asyncio_demo.py` | a blocking call in `async def` vs `to_thread`, 3 users | 6 s vs 2 s |
| `asyncio_limits_demo.py` | 10 paid tasks, service limit 3 | unbounded `gather`: 429 errors and charges for abandoned tasks; estimate + semaphore + `TaskGroup`: 4 s, 10 of 10 |
| `checkpoint_demo.py` | a crash in the middle of 10 paid steps | without a checkpoint 17 charges, with one 11: the step in flight at the crash is why you need an idempotency key |
| `parallel_bench.py` | the golden set through the full RAG on local ollama | a client with concurrency 8 is ~1.1×: the server is the bottleneck; ollama batching is ~1.2× for +1.4 GB |

## Limitations

What not to trust in these numbers:

- The golden set is small (22 queries). Differences under ~0.05 are noise.
- The fusion weight was tuned without a held-out test set — that is overfitting
  to the eval set. The right way: split dev/test up front.
- The `structured` category stays at 0.67 at k=3. The auto-filter lifts it to
  1.00 at k=5, but gives nothing at the working point k=3 and costs 2–5 s, so it
  is off by default.
- groundedness is entity-level and blind to invented claims (claim-level).
- The refusal metric counts as a refusal an answer that starts with "no data"
  but then answers from general knowledge anyway.
- There is no corpus of relations between types.
- The agent measurements are small (6 control questions, 16 in the tool-design
  eval, one model): they show the effect size on a failure class, not a share
  of real traffic.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama serve &
ollama pull bge-m3 && ollama pull qwen2.5:7b

.venv/bin/python src/ingest.py all          # build all collections
.venv/bin/python src/rag.py "кто такой Дон Кихот"
.venv/bin/python src/evaluate.py            # compare retrieval configurations
.venv/bin/python src/eval_generation.py     # compare generation prompts
.venv/bin/python src/sweep.py bge-m3 chunked  # fusion weight sweep

# MCP: check the server over the raw protocol and run the tests
.venv/bin/python src/mcp_raw_client.py
.venv/bin/python -m pytest tests

# Python agent and agent evaluations
.venv/bin/python src/agent.py
.venv/bin/python src/eval_tool_design.py --orders
.venv/bin/python src/eval_people.py

# TypeScript: client and agent (Node 22.18+ runs .ts directly)
cd ts && npm install
node mcp_client.ts && node agent.ts
npx tsc && node --test

# Telegram bot (get a token from @BotFather, never commit it)
export TELEGRAM_BOT_TOKEN="..."
.venv/bin/python src/bot.py
```

## Layout

| File | Purpose |
|---|---|
| `config.py` | every experiment knob in one place |
| `documents.py` | corpus assembly: whole types and chunks |
| `embeddings.py` | provider-agnostic embedding layer |
| `lexical.py` | BM25 with a truncation stemmer |
| `retrieve.py` | three search modes + small-to-big |
| `ingest.py` | idempotent loader |
| `prompts.py` | system prompt variants |
| `generate.py` | provider-agnostic LLM wrapper |
| `query_understanding.py` | structural query parsing with a rule-based gatekeeper |
| `rag.py` | the full pipeline: search → prompt → answer |
| `bot.py` | Telegram interface; the blocking RAG call goes to a thread |
| `mcp_server.py` | MCP server: tools, resources, prompts, graceful degradation |
| `mcp_raw_client.py` | client without the SDK, shows JSON-RPC line by line |
| `tests/` | invariants for degradation, timeouts and `filter_types` |
| `evaluate.py` | retrieval configuration bench |
| `eval_generation.py` | groundedness, citations and refusals |
| `sweep.py` | fusion hyperparameter sweep |
| `agent.py` | agent loop: repair of malformed calls, step limit |
| `eval_tool_design.py` | tool parameter shape: boolean vs enum, all orders |
| `eval_people.py` | the agent and real people's names + counter-check on aliases |
| `asyncio_demo.py`, `asyncio_limits_demo.py`, `checkpoint_demo.py`, `parallel_bench.py` | asyncio, limits, idempotency, concurrency measurements |
| `ts/mcp_client.ts` | TypeScript MCP client (SDK) for the Python server |
| `ts/agent_loop.ts` | TypeScript agent loop: error routing, allowlist, no-progress |
| `ts/agent.ts` | the reference agent on top of the loop |
| `ts/agent_loop.test.ts` | loop mechanics tests (`node --test`) |

## License

[MIT](LICENSE)
