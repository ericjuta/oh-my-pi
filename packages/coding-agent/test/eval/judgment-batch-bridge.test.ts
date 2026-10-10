import { afterEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import * as vm from "node:vm";
import type { Api, AssistantMessage, Model } from "@oh-my-pi/pi-ai";
import * as ai from "@oh-my-pi/pi-ai";
import { TempDir } from "@oh-my-pi/pi-utils";
import { $ } from "bun";
import { AsyncJobManager } from "../../src/async";
import { ModelRegistry } from "../../src/config/model-registry";
import { Settings } from "../../src/config/settings";
import {
	type JudgmentBatchItem,
	type JudgmentBatchStatus,
	releaseJudgmentBatches,
	runEvalJudgmentBatch,
} from "../../src/eval/judgment-batch-bridge";
import { JAVASCRIPT_PRELUDE_SOURCE } from "../../src/eval/js/shared/prelude";
import {
	isJudgmentBatchProgress,
	JUDGMENT_BATCH_PROGRESS_EVENT_CHANNEL,
	type JudgmentBatchProgress,
} from "../../src/eval/judgment-batch-events";
import { EventBus } from "../../src/utils/event-bus";
import type { PythonResult } from "../../src/eval/py/executor";
import type { ToolSession } from "../../src/tools";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";
import { asGlobalFetch } from "../helpers/fetch-mock";

const SMOL: Model<Api> = {
	id: "smol",
	name: "smol",
	api: "openai-responses",
	provider: "p",
	baseUrl: "https://example.test/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 1 },
	contextWindow: 128000,
	maxTokens: 4096,
} as Model<Api>;

const QUESTIONS = {
	tests: { type: "bool", instructions: "Does the request mention tests?" },
};

const managers = new Set<AsyncJobManager>();

interface BatchSession {
	session: ToolSession;
	manager?: AsyncJobManager;
}

function makeSession(opts: { agentId?: string; jobs?: boolean } = {}): BatchSession {
	const settings = Settings.isolated({
		"async.enabled": false,
		"task.isolation.enabled": false,
		modelRoles: { judge: "p/smol" },
	});
	const authStorage = createInMemoryAuthStorage();
	authStorage.keys.setRuntime("p", "test-key");
	const modelRegistry = new ModelRegistry(authStorage, "/nonexistent/judgment-batch-models.yml");
	vi.spyOn(modelRegistry, "getAvailable").mockReturnValue([SMOL]);
	const session: Record<string, unknown> = {
		settings,
		modelRegistry,
		getSessionId: () => "sess-1",
		getAgentId: () => opts.agentId ?? "Main",
	};
	if (opts.jobs === false) return { session: session as unknown as ToolSession };
	const manager = new AsyncJobManager({
		retentionMs: 60_000,
		onJobComplete: () => {},
	});
	managers.add(manager);
	session.asyncJobManager = manager;
	return { session: session as unknown as ToolSession, manager };
}

/** Billed amount per mocked judgment attempt; batches sum it into `status().cost`. */
const REPLY_COST = 0.001;

function reply(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-responses",
		provider: "p",
		model: "smol",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				total: REPLY_COST,
			},
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

/** Route each completion by the state text embedded in the prompt; gated states block until released. */
function mockJudge(answers: Record<string, string | Promise<string>>): void {
	vi.spyOn(ai, "completeSimple").mockImplementation(async (_model, context, options) => {
		const user = context.messages.find(message => message.role === "user");
		const prompt = typeof user?.content === "string" ? user.content : JSON.stringify(user?.content);
		let message = reply("tests: no");
		for (const state in answers) {
			if (prompt.includes(state)) {
				message = reply(await answers[state]);
				break;
			}
		}
		options?.onAttempt?.(message);
		return message;
	});
}

async function create(session: ToolSession, states: unknown[], extra: Record<string, unknown> = {}) {
	const items = states.map((state, key) => ({ key, state }));
	return (await runEvalJudgmentBatch(
		{ op: "create", items, questions: QUESTIONS, ...extra },
		{ session },
	)) as JudgmentBatchStatus;
}

async function drain(session: ToolSession, id: string, timeoutMs?: number): Promise<JudgmentBatchItem[]> {
	const result = (await runEvalJudgmentBatch({ op: "drain", id, timeoutMs }, { session })) as {
		items: JudgmentBatchItem[];
	};
	return result.items;
}

async function status(session: ToolSession, id: string): Promise<JudgmentBatchStatus> {
	return (await runEvalJudgmentBatch({ op: "status", id }, { session })) as JudgmentBatchStatus;
}

afterEach(async () => {
	releaseJudgmentBatches("Main");
	releaseJudgmentBatches("Other");
	for (const manager of managers) await manager.dispose();
	managers.clear();
	vi.restoreAllMocks();
});

describe("judge_batch bridge", () => {
	it("publishes live progress between cells without a job manager or drain calls", async () => {
		const gate = Promise.withResolvers<string>();
		mockJudge({ "state-fast": "tests: yes", "state-gated": gate.promise });
		const { session } = makeSession({ jobs: false });
		session.eventBus = new EventBus();
		const events: JudgmentBatchProgress[] = [];
		const intermediate = Promise.withResolvers<void>();
		const finished = Promise.withResolvers<void>();
		session.eventBus.on(JUDGMENT_BATCH_PROGRESS_EVENT_CHANNEL, event => {
			if (!isJudgmentBatchProgress(event)) return;
			events.push(event);
			if (event.running && event.done === 1) intermediate.resolve();
			if (!event.running) finished.resolve();
		});
		try {
			await create(session, ["state-fast", "state-gated"], {
				intent: "Classifying: Test idiomacy",
			});
			await intermediate.promise;
			expect(events[0]).toMatchObject({
				intent: "Classifying: Test idiomacy",
				done: 0,
				total: 2,
				cost: 0,
				running: true,
			});
			expect(events.at(-1)).toMatchObject({ done: 1, total: 2, running: true });
		} finally {
			gate.resolve("tests: no");
		}
		await finished.promise;
		expect(events.at(-1)).toMatchObject({
			done: 2,
			total: 2,
			failed: 0,
			running: false,
		});
		expect(events.at(-1)?.cost).toBeCloseTo(2 * REPLY_COST, 6);
	});
	it("validates items and options before starting", async () => {
		const { session } = makeSession();
		const spy = vi.spyOn(ai, "completeSimple");
		await expect(
			runEvalJudgmentBatch({ op: "create", items: [], questions: QUESTIONS }, { session }),
		).rejects.toThrow("items must not be empty");
		await expect(
			runEvalJudgmentBatch(
				{
					op: "create",
					items: [
						{ key: "a", state: "x" },
						{ key: "a", state: "y" },
					],
					questions: QUESTIONS,
				},
				{ session },
			),
		).rejects.toThrow('duplicate item key "a"');
		await expect(
			runEvalJudgmentBatch(
				{
					op: "create",
					items: [{ key: "a", state: "" }],
					questions: QUESTIONS,
				},
				{ session },
			),
		).rejects.toThrow("state must not be empty");
		await expect(
			runEvalJudgmentBatch(
				{
					op: "create",
					items: [{ key: "a", state: "x" }],
					questions: QUESTIONS,
					concurrency: -1,
				},
				{ session },
			),
		).rejects.toThrow("concurrency must be a non-negative integer");
		await expect(runEvalJudgmentBatch({ op: "drain", id: "jdgb-missing" }, { session })).rejects.toThrow(
			'unknown judge_batch "jdgb-missing"',
		);
		expect(spy).not.toHaveBeenCalled();
	});

	it("drains settled items through a cursor, records failures per item, and settles the job", async () => {
		const gate = Promise.withResolvers<string>();
		mockJudge({
			"state-a": "tests: yes",
			"state-b": "I cannot decide.",
			"state-c": gate.promise,
		});
		const { session, manager } = makeSession();
		const created = await create(session, ["state-a", "state-b", "state-c"], {
			retries: 0,
		});
		expect(created.total).toBe(3);
		expect(created.running).toBe(true);

		// Near-simultaneous settles coalesce into one drain instead of one round-trip each.
		const first = await drain(session, created.id, 5_000);
		expect(first.map(item => item.key).sort()).toEqual([0, 1]);
		expect(first.find(item => item.key === 0)).toEqual({
			key: 0,
			answers: { tests: { type: "bool", bool: 1 } },
			model: "p/smol",
			selectedModel: "p/smol",
			primaryModel: "p/smol",
			responseModel: undefined,
			fallback: false,
			cache: "miss",
			questions: 1,
			cachedQuestions: 0,
		});
		const failedItem = first.find(item => item.key === 1);
		expect(failedItem?.error).toContain('judgment "tests"');
		// Failed parsing must not reuse the preceding successful item's metadata.
		expect(Object.keys(failedItem!).sort()).toEqual(["error", "key"]);
		// Nothing new and the run is still going: a zero timeout returns immediately with nothing.
		expect(await drain(session, created.id, 0)).toEqual([]);

		const pending = drain(session, created.id, 5_000);
		gate.resolve("tests: no");
		expect(await pending).toEqual([
			{
				key: 2,
				answers: { tests: { type: "bool", bool: 0 } },
				model: "p/smol",
				selectedModel: "p/smol",
				primaryModel: "p/smol",
				responseModel: undefined,
				fallback: false,
				cache: "miss",
				questions: 1,
				cachedQuestions: 0,
			},
		]);
		expect(await drain(session, created.id, 0)).toEqual([]);

		const final = await status(session, created.id);
		expect(final).toMatchObject({
			done: 3,
			failed: 1,
			running: false,
			model: "p/smol",
		});
		expect(final.error).toBeUndefined();
		expect(await runEvalJudgmentBatch({ op: "results", id: created.id }, { session })).toEqual({
			results: {
				"0": { tests: { type: "bool", bool: 1 } },
				"2": { tests: { type: "bool", bool: 0 } },
			},
		});
		const failed = (await runEvalJudgmentBatch({ op: "failed", id: created.id }, { session })) as {
			failed: Record<string, string>;
		};
		expect(Object.keys(failed.failed)).toEqual(["1"]);

		const job = manager?.getJob(created.id);
		expect(job?.status).toBe("completed");
		// Five attempts: a, three parse retries on b, c — cost counts every attempt, not settled items.
		expect(job?.resultText).toMatch(/^judged 3\/3 · 1 failed · \$0\.0050 · p\/smol · [\d.]+s$/);
	});

	it("retries an item once before recording its failure", async () => {
		let attempts = 0;
		vi.spyOn(ai, "completeSimple").mockImplementation(async () => {
			attempts++;
			return reply(attempts <= 3 ? "I cannot decide." : "tests: yes");
		});
		const { session } = makeSession();
		const created = await create(session, ["state-a"]);
		const [item] = await drain(session, created.id, 5_000);
		expect(item?.answers).toEqual({ tests: { type: "bool", bool: 1 } });
		// Three off-format turns exhaust the first judgment; the retry answers on its first turn.
		expect(attempts).toBe(4);
	});

	it("raises from drain only after the cursor is exhausted when min_ok is unmet", async () => {
		mockJudge({ "state-a": "I cannot decide." });
		const { session, manager } = makeSession();
		const created = await create(session, ["state-a"], {
			retries: 0,
			minOk: 1,
		});
		const items = await drain(session, created.id, 5_000);
		expect(items).toHaveLength(1);
		expect(items[0]?.error).toContain('judgment "tests"');
		await expect(drain(session, created.id, 0)).rejects.toThrow("only 0/1 item(s) judged (min_ok=1)");
		expect(manager?.getJob(created.id)?.status).toBe("failed");
	});

	it("cancels in-flight and unstarted items and cancels the job", async () => {
		const gate = Promise.withResolvers<string>();
		mockJudge({ "state-a": gate.promise, "state-b": gate.promise });
		const { session, manager } = makeSession();
		const created = await create(session, ["state-a", "state-b"], {
			concurrency: 1,
		});
		expect(await drain(session, created.id, 0)).toEqual([]);

		expect(await runEvalJudgmentBatch({ op: "cancel", id: created.id }, { session })).toEqual({ cancelled: true });
		gate.resolve("tests: yes");
		const items = await drain(session, created.id, 5_000);
		expect(items).toEqual([
			{ key: 0, error: "cancelled" },
			{ key: 1, error: "cancelled" },
		]);
		expect((await status(session, created.id)).running).toBe(false);
		expect(await runEvalJudgmentBatch({ op: "cancel", id: created.id }, { session })).toEqual({ cancelled: false });
		expect(manager?.getJob(created.id)?.status).toBe("cancelled");
	});

	it("scopes attach and release to the owning agent", async () => {
		mockJudge({ "state-a": "tests: yes" });
		const { session } = makeSession();
		const created = await create(session, ["state-a"]);
		const attached = (await runEvalJudgmentBatch(
			{ op: "attach", id: created.id },
			{ session },
		)) as JudgmentBatchStatus;
		expect(attached.id).toBe(created.id);
		expect(attached.total).toBe(1);
		await expect(
			runEvalJudgmentBatch({ op: "attach", id: created.id }, { session: makeSession({ agentId: "Other" }).session }),
		).rejects.toThrow(`unknown judge_batch "${created.id}"`);

		releaseJudgmentBatches("Main");
		await expect(runEvalJudgmentBatch({ op: "attach", id: created.id }, { session })).rejects.toThrow(
			`unknown judge_batch "${created.id}"`,
		);
	});

	it("runs without a job manager", async () => {
		mockJudge({ "state-a": "tests: yes" });
		const { session } = makeSession({ jobs: false });
		const created = await create(session, ["state-a"]);
		expect(await drain(session, created.id, 5_000)).toEqual([
			{
				key: 0,
				answers: { tests: { type: "bool", bool: 1 } },
				model: "p/smol",
				selectedModel: "p/smol",
				primaryModel: "p/smol",
				responseModel: undefined,
				fallback: false,
				cache: "miss",
				questions: 1,
				cachedQuestions: 0,
			},
		]);
		expect((await status(session, created.id)).running).toBe(false);
	});
});

describe("judgeBatch() JS prelude", () => {
	function hostSandbox(session: ToolSession): vm.Context {
		const sandbox = vm.createContext({
			questions: QUESTIONS,
			__omp_call_tool__: async (name: string, args: Record<string, unknown>) => {
				if (name !== "__judge_batch__") throw new Error(`unexpected bridge call ${name}`);
				return runEvalJudgmentBatch(args, { session });
			},
		});
		vm.runInContext(JAVASCRIPT_PRELUDE_SOURCE, sandbox);
		return sandbox;
	}

	it("preserves native provider provenance after VM reattach without replaying the shared drain", async () => {
		const { session } = makeSession({ jobs: false });
		vi.spyOn(session.modelRegistry!, "getAvailable").mockReturnValue([{ ...SMOL, api: "typesafe" }]);
		const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(
			asGlobalFetch(async (_url, init) => {
				const request = JSON.parse(String(init?.body));
				expect(request.model).toBe("smol");
				expect(request.questions).toEqual({
					tests: { type: "noul", instructions: QUESTIONS.tests.instructions },
				});
				return Response.json({
					model: "served-tests-revision",
					answers: { tests: { type: "noul", noul: 0.83 } },
					usage: { input_tokens: 10, output_tokens: 1 },
				});
			}),
		);
		const original = hostSandbox(session);
		original.states = { a: ["js-native-provenance", crypto.randomUUID()] };
		const id = await vm.runInContext(
			`(async () => {
				globalThis.batch = await judgeBatch(states, questions);
				return batch.id;
			})()`,
			original,
		);
		const restored = hostSandbox(session);
		restored.id = id;
		const seen = await vm.runInContext(
			`(async () => {
				const batch = await judgeBatch.attach(id);
				const items = await batch.drain({ timeout: 5 });
				return {
					id: batch.id,
					items: items.map(([key, item]) => [key, { ...item, ok: item.ok }]),
					results: await batch.results(),
				};
			})()`,
			restored,
		);
		expect(seen).toStrictEqual({
			id,
			items: [
				[
					"a",
					{
						key: "a",
						answers: { tests: { type: "bool", bool: 0.83 } },
						error: undefined,
						model: "p/served-tests-revision",
						selectedModel: "p/smol",
						primaryModel: "p/smol",
						responseModel: "served-tests-revision",
						fallback: false,
						cache: "miss",
						questions: 1,
						cachedQuestions: 0,
						ok: true,
					},
				],
			],
			results: { a: { tests: { type: "bool", bool: 0.83 } } },
		});
		expect(await vm.runInContext("batch.drain({ timeout: 0 })", original)).toEqual([]);
		expect(await vm.runInContext("batch.results()", original)).toEqual(seen.results);
		expect(fetch).toHaveBeenCalledTimes(1);
	});

	it("keeps unavailable response identity and failed-item provenance undefined through the real host", async () => {
		vi.spyOn(ai, "completeSimple").mockImplementation(async (_model, context, options) => {
			const user = context.messages.find(message => message.role === "user");
			const prompt = typeof user?.content === "string" ? user.content : JSON.stringify(user?.content);
			const message = reply("tests: yes");
			if (prompt.includes("js-provider-failure")) {
				message.stopReason = "error";
				message.errorMessage = "js-provider-failure";
			}
			options?.onAttempt?.(message);
			return message;
		});
		const { session } = makeSession({ jobs: false });
		const sandbox = hostSandbox(session);
		const seen = await vm.runInContext(
			`(async () => {
				const batch = await judgeBatch(
					{ good: "js-no-response-identity", bad: "js-provider-failure" },
					questions,
					{ retries: 0, concurrency: 1 },
				);
				const items = {};
				for await (const [key, item] of batch.drainIter({ timeout: 5 })) {
					items[key] = { ...item, ok: item.ok };
				}
				return { items, results: await batch.results(), failed: await batch.failed() };
			})()`,
			sandbox,
		);
		expect(seen.items.good).toStrictEqual({
			key: "good",
			answers: { tests: { type: "bool", bool: 1 } },
			error: undefined,
			model: "p/smol",
			selectedModel: "p/smol",
			primaryModel: "p/smol",
			responseModel: undefined,
			fallback: false,
			cache: "miss",
			questions: 1,
			cachedQuestions: 0,
			ok: true,
		});
		expect(seen.items.bad.error).toContain("js-provider-failure");
		expect(seen.items.bad).toStrictEqual({
			key: "bad",
			answers: undefined,
			error: seen.items.bad.error,
			model: undefined,
			selectedModel: undefined,
			primaryModel: undefined,
			responseModel: undefined,
			fallback: undefined,
			cache: undefined,
			questions: undefined,
			cachedQuestions: undefined,
			ok: false,
		});
		expect(seen.results).toEqual({ good: { tests: { type: "bool", bool: 1 } } });
		expect(seen.failed).toEqual({ bad: seen.items.bad.error });
	});

	it("maps states to keyed items and drainIter stops on an empty drain", async () => {
		const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
		const drains = [
			{
				items: [
					{
						key: "a",
						answers: { ok: { type: "bool", bool: 1 } },
						model: "p/smol",
					},
				],
			},
			{ items: [{ key: "b", error: "boom" }] },
			{ items: [] },
		];
		const sandbox: Record<string, unknown> = {
			__omp_call_tool__: async (name: string, args: Record<string, unknown>) => {
				calls.push({ name, args });
				if (name !== "__judge_batch__") throw new Error(`unexpected bridge call ${name}`);
				if (args.op === "create") return { id: "jdgb-1", total: 2 };
				if (args.op === "drain") return drains.shift();
				if (args.op === "status") return { id: "jdgb-1", done: 2, total: 2, failed: 1, running: false };
				throw new Error(`unexpected op ${String(args.op)}`);
			},
		};
		vm.createContext(sandbox);
		vm.runInContext(JAVASCRIPT_PRELUDE_SOURCE, sandbox);

		const seen = await vm.runInContext(
			`(async () => {
				const b = await judgeBatch({ a: "first", b: "second" }, { ok: { type: "bool", instructions: "?" } }, { concurrency: 4 });
				const out = [];
				for await (const [key, item] of b.drainIter({ timeout: 30 })) out.push([key, item.ok, item.answers ?? item.error]);
				return { id: b.id, total: b.total, out, status: await b.status() };
			})()`,
			sandbox,
		);

		expect(seen).toEqual({
			id: "jdgb-1",
			total: 2,
			out: [
				["a", true, { ok: { type: "bool", bool: 1 } }],
				["b", false, "boom"],
			],
			status: { id: "jdgb-1", done: 2, total: 2, failed: 1, running: false },
		});
		expect(calls[0]?.args).toEqual({
			op: "create",
			items: [
				{ key: "a", state: "first" },
				{ key: "b", state: "second" },
			],
			questions: { ok: { type: "bool", instructions: "?" } },
			concurrency: 4,
		});
		const drainCalls = calls.filter(call => call.args.op === "drain");
		expect(drainCalls).toHaveLength(3);
		expect(drainCalls[0]?.args.id).toBe("jdgb-1");
		expect(drainCalls[0]?.args.timeoutMs).toBeLessThanOrEqual(30_000);
	});
});

interface PythonJudgeBatchResult extends PythonResult {
	/** Native wire requests the host sent, as `[state, questionIds]`. */
	requests: Array<[string, string[]]>;
}

async function runPythonJudgeBatchInSubprocess(tempDir: TempDir): Promise<PythonJudgeBatchResult> {
	const repoRoot = path.resolve(import.meta.dir, "../../..");
	const scriptPath = path.join(tempDir.path(), "run-python-judge-batch.ts");
	const resultPath = path.join(tempDir.path(), "python-judge-batch-result.json");
	const executorPath = path.resolve(import.meta.dir, "../../src/eval/py/executor.ts");
	const settingsPath = path.resolve(import.meta.dir, "../../src/config/settings.ts");
	const registryPath = path.resolve(import.meta.dir, "../../src/config/model-registry.ts");
	const setupPath = path.resolve(import.meta.dir, "../helpers/agent-session-setup.ts");
	// stdout is presentation-capped per line; display(dict) preserves structured JSON evidence.
	const cellOne = [
		'Q = {"tests": {"type": "bool", "instructions": "Does the request mention tests?"}}',
		'b = judge_batch({"a": "add tests please", "b": "rename a local", "c": "provider failure"}, Q, retries=0, intent="Classifying: Test idiomacy")',
		"items = sorted([entry async for entry in b.drain_iter(timeout=30)], key=lambda kv: kv[0])",
		'fields = ("selected_model", "primary_model", "response_model", "fallback", "cache", "questions", "cached_questions")',
		"def provenance(item): return {field: getattr(item, field) for field in fields}",
		"single = await judge('write tests', Q)",
		'absent = JudgmentItem({"answers": {}})',
		'display({"id": b.id, "total": b.total, "items": [[k, i.ok, i.answers or i.error, provenance(i)] for k, i in items], "single": single, "absent": provenance(absent)})',
	].join("\n");
	const cellTwo = [
		"again = judge_batch.attach(b.id)",
		"rest = await again.drain(timeout=0)",
		'attached = {"rest": rest, "status": again.status()["done"], "results": sorted(again.results()), "failed": sorted(again.failed())}',
		"again.close()",
		'Q2 = {**Q, "short": {"type": "bool", "instructions": "Is the request short?"}}',
		'partial = judge_batch({"partial": "add tests please"}, Q2)',
		'hits = judge_batch({"hit": "add tests please", "unknown": "rename a local"}, Q)',
		"cached = [entry async for entry in partial.drain_iter(timeout=30)] + [entry async for entry in hits.drain_iter(timeout=30)]",
		'display({**attached, "cached": [[k, i.answers, provenance(i)] for k, i in sorted(cached, key=lambda kv: kv[0])]})',
	].join("\n");
	await Bun.write(
		scriptPath,
		`
import { vi } from "bun:test";
import { executePython } from ${JSON.stringify(executorPath)};
import { Settings } from ${JSON.stringify(settingsPath)};
import { ModelRegistry } from ${JSON.stringify(registryPath)};
import { createInMemoryAuthStorage } from ${JSON.stringify(setupPath)};

const SMOL = { ...${JSON.stringify(SMOL)}, api: "typesafe" };
const settings = Settings.isolated({ "async.enabled": false, "task.isolation.enabled": false, modelRoles: { judge: "p/smol" } });
const authStorage = createInMemoryAuthStorage();
authStorage.keys.setRuntime("p", "test-key");
const modelRegistry = new ModelRegistry(authStorage, "/nonexistent/judgment-batch-py-models.yml");
vi.spyOn(modelRegistry, "getAvailable").mockReturnValue([SMOL]);
const session = { cwd: ${JSON.stringify(tempDir.path())}, settings, modelRegistry, getSessionId: () => "sess-py", getAgentId: () => "Main" };
const requests = [];
vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
	if (String(input) !== "https://example.test/v1/v1/systemone") throw new Error("unexpected judgment endpoint");
	const request = JSON.parse(String(init?.body));
	if (request.model !== "smol") throw new Error("unexpected requested model");
	requests.push([request.state, Object.keys(request.questions)]);
	if (request.state === "provider failure") return new Response("fixture provider failure", { status: 400 });
	const bool = request.state.includes("tests") ? 1 : 0;
	return Response.json({
		...(bool ? { model: "served-tests" } : {}),
		answers: Object.fromEntries(Object.keys(request.questions).map(id => [id, { type: "noul", noul: bool }])),
		usage: { input_tokens: 3, output_tokens: 1 },
	});
});
const options = {
	cwd: ${JSON.stringify(tempDir.path())},
	sessionId: "py-judge-batch",
	sessionFile: ${JSON.stringify(path.join(tempDir.path(), "session.jsonl"))},
	toolSession: session,
};
const one = await executePython(${JSON.stringify(cellOne)}, options);
const two = await executePython(${JSON.stringify(cellTwo)}, options);
await Bun.write(${JSON.stringify(resultPath)}, JSON.stringify({ exitCode: one.exitCode || two.exitCode, output: one.output + "\\n" + two.output, displayOutputs: [...one.displayOutputs, ...two.displayOutputs], requests }));
process.exit(0);
`,
	);
	// The script is not a test entrypoint, so the shared judgment cache is live; keep it in the temp dir.
	const child = await $`bun ${scriptPath}`
		.cwd(repoRoot)
		.env({ ...process.env, OMP_JUDGMENT_CACHE_DB: path.join(tempDir.path(), "judgment-cache.db") })
		.quiet()
		.nothrow();
	const stdout = child.stdout.toString();
	const stderr = child.stderr.toString();
	if (child.exitCode !== 0) {
		throw new Error(
			`Python judge_batch subprocess exited with ${child.exitCode}\nstdout:\n${stdout}\nstderr:\n${stderr}`,
		);
	}
	const result = (await Bun.file(resultPath).json()) as PythonJudgeBatchResult;
	if (result.exitCode !== 0) {
		throw new Error(
			`Python judge_batch cell exited with ${result.exitCode}\nPython output:\n${result.output}\nstdout:\n${stdout}\nstderr:\n${stderr}`,
		);
	}
	return result;
}

describe("judge_batch() Python prelude", () => {
	it("preserves native and cached provenance, keeps missing values None, and leaves judge answers and the shared drain cursor unchanged", async () => {
		const tempDir = TempDir.createSync("@omp-eval-judge-batch-py-");
		try {
			const result = await runPythonJudgeBatchInSubprocess(tempDir);
			expect(result.exitCode).toBe(0);
			const evidence = result.displayOutputs.flatMap(output => (output.type === "json" ? [output.data] : []));
			expect(evidence).toHaveLength(2);
			const [one, two] = evidence;
			const absent = {
				selected_model: null,
				primary_model: null,
				response_model: null,
				fallback: null,
				cache: null,
				questions: null,
				cached_questions: null,
			};
			const routed = {
				selected_model: "p/smol",
				primary_model: "p/smol",
				response_model: "served-tests",
				fallback: false,
				cache: "miss",
				questions: 1,
				cached_questions: 0,
			};
			expect(one).toEqual({
				id: expect.stringMatching(/^jdgb-/),
				total: 3,
				items: [
					["a", true, { tests: { type: "bool", bool: 1 } }, routed],
					["b", true, { tests: { type: "bool", bool: 0 } }, { ...routed, response_model: null }],
					["c", false, expect.stringContaining("fixture provider failure"), absent],
				],
				single: { tests: { type: "bool", bool: 1 } },
				absent,
			});
			expect(two).toEqual({
				rest: [],
				status: 3,
				results: ["a", "b"],
				failed: ["c"],
				cached: [
					["hit", { tests: { type: "bool", bool: 1 } }, { ...routed, cache: "hit", cached_questions: 1 }],
					[
						"partial",
						{ tests: { type: "bool", bool: 1 }, short: { type: "bool", bool: 1 } },
						{ ...routed, cache: "partial", questions: 2, cached_questions: 1 },
					],
					[
						"unknown",
						{ tests: { type: "bool", bool: 0 } },
						{ ...routed, response_model: null, cache: "hit", cached_questions: 1 },
					],
				],
			});
			// Cached answers never reach the provider: only the partial item's uncached question is sent.
			expect(
				result.requests.toSorted((x, y) => x[0].localeCompare(y[0]) || x[1].join().localeCompare(y[1].join())),
			).toEqual([
				["add tests please", ["short"]],
				["add tests please", ["tests"]],
				["provider failure", ["tests"]],
				["rename a local", ["tests"]],
				["write tests", ["tests"]],
			]);
		} finally {
			tempDir.removeSync();
		}
	});
});
