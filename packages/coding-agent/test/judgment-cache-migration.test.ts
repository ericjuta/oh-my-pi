import { Database } from "bun:sqlite";
import { describe, expect, it } from "bun:test";
import * as path from "node:path";
import type { JudgmentRequest, JudgmentResult } from "@oh-my-pi/pi-ai";
import { JudgmentCache } from "@oh-my-pi/pi-coding-agent/judgment/cache";
import { TempDir } from "@oh-my-pi/pi-utils";

const MODEL = "typesafe/jev-latest";
const STATE = { body: "legacy body", file: "legacy.ts" };
const STATE_JSON = '{"body":"legacy body","file":"legacy.ts"}';
const STATE_ID = new Bun.CryptoHasher("sha256").update(STATE_JSON).digest("hex");
const LEGACY_REQUEST = {
	state: STATE,
	questions: {
		relevant: { type: "noul", instructions: "Is this relevant?" },
		safe: { type: "noul", instructions: "Is this safe?", criteria: { true: "Safe to retain" } },
	},
} satisfies JudgmentRequest;
const LEGACY_ANSWERS = {
	relevant: { type: "noul", noul: 0.875 },
	safe: { type: "noul", noul: 0.25 },
} satisfies JudgmentResult<typeof LEGACY_REQUEST.questions>["answers"];
const LEGACY_ORACLE = [
	{
		id: 7,
		state: STATE_ID,
		model: MODEL,
		name: "relevant",
		type: "noul",
		instruction: "Is this relevant?",
		criteria: "null",
		answer: '{"type":"noul","noul":0.875}',
		created_at: 1_700_000_001,
	},
	{
		id: 11,
		state: STATE_ID,
		model: MODEL,
		name: "safe",
		type: "noul",
		instruction: "Is this safe?",
		criteria: '{"true":"Safe to retain"}',
		answer: '{"type":"noul","noul":0.25}',
		created_at: 1_700_000_002,
	},
];
const LEGACY_USAGE = [
	{
		id: 13,
		state: STATE_ID,
		results: "[7,11]",
		provider: "typesafe",
		model: "legacy-compatibility-label",
		input_tokens: 120,
		output_tokens: 8,
		price: 0.125,
		created_at: 1_700_000_003,
	},
	{
		id: 17,
		state: STATE_ID,
		results: "[11]",
		provider: "typesafe",
		model: "another-compatibility-label",
		input_tokens: 240,
		output_tokens: 16,
		price: 0.25,
		created_at: 1_700_000_004,
	},
];
const FRESH_REQUEST = {
	state: STATE,
	questions: { fresh: { type: "noul", instructions: "Is this newly evaluated?" } },
} satisfies JudgmentRequest;
const FRESH_RESULT = {
	api: "typesafe",
	provider: "typesafe",
	model: "new-compatibility-label",
	responseModel: "jev-raw-origin-v2",
	answers: { fresh: { type: "noul", noul: 0.625 } },
	usage: {
		input: 37,
		output: 5,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 42,
		cost: { input: 0.0625, output: 0, cacheRead: 0, cacheWrite: 0, total: 0.0625 },
	},
} satisfies JudgmentResult<typeof FRESH_REQUEST.questions>;

function seedLegacyDatabase(dbPath: string): void {
	using db = new Database(dbPath);
	// This is the pre-provenance schema, not a current cache with its data edited afterward.
	db.run(`
		PRAGMA journal_mode=WAL;
		PRAGMA synchronous=NORMAL;
		PRAGMA foreign_keys=ON;
		CREATE TABLE states (id TEXT PRIMARY KEY, state TEXT NOT NULL, created_at INTEGER NOT NULL);
		CREATE TABLE oracle (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			state TEXT NOT NULL REFERENCES states(id),
			model TEXT NOT NULL,
			name TEXT NOT NULL,
			type TEXT NOT NULL,
			instruction TEXT NOT NULL,
			criteria TEXT NOT NULL,
			answer TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			UNIQUE (state, model, name, type, instruction, criteria)
		);
		CREATE TABLE usage (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			state TEXT NOT NULL REFERENCES states(id),
			results TEXT NOT NULL,
			provider TEXT NOT NULL,
			model TEXT NOT NULL,
			input_tokens INTEGER NOT NULL,
			output_tokens INTEGER NOT NULL,
			price REAL NOT NULL,
			created_at INTEGER NOT NULL
		);
		CREATE INDEX idx_usage_state ON usage(state);
	`);
	db.query("INSERT INTO states (id, state, created_at) VALUES (?, ?, ?)").run(STATE_ID, STATE_JSON, 1_700_000_000);
	const insertOracle = db.query(
		"INSERT INTO oracle (id, state, model, name, type, instruction, criteria, answer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
	);
	for (const row of LEGACY_ORACLE) {
		insertOracle.run(
			row.id,
			row.state,
			row.model,
			row.name,
			row.type,
			row.instruction,
			row.criteria,
			row.answer,
			row.created_at,
		);
	}
	const insertUsage = db.query(
		"INSERT INTO usage (id, state, results, provider, model, input_tokens, output_tokens, price, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
	);
	for (const row of LEGACY_USAGE) {
		insertUsage.run(
			row.id,
			row.state,
			row.results,
			row.provider,
			row.model,
			row.input_tokens,
			row.output_tokens,
			row.price,
			row.created_at,
		);
	}
	expect(
		db
			.query<{ name: string }, []>("PRAGMA table_info(oracle)")
			.all()
			.map(column => column.name),
	).not.toContain("response_model");
}

function expectLegacyLookup(cache: JudgmentCache): void {
	const hit = cache.lookup(MODEL, LEGACY_REQUEST);
	expect(hit.stateId).toBe(STATE_ID);
	expect(hit.state).toBe(STATE_JSON);
	expect(hit.answers).toEqual(LEGACY_ANSWERS);
	expect(hit.responseModel).toBeUndefined();
}

function expectPersistedRows(dbPath: string, hasFreshRecord = false): void {
	using db = new Database(dbPath, { readonly: true });
	const columns = db
		.query<{ name: string; type: string; notnull: number; dflt_value: string | null }, []>(
			"PRAGMA table_info(oracle)",
		)
		.all()
		.filter(column => column.name === "response_model")
		.map(({ name, type, notnull, dflt_value }) => ({ name, type, notnull, dflt_value }));
	expect(columns).toEqual([{ name: "response_model", type: "TEXT", notnull: 0, dflt_value: null }]);
	expect(db.query("SELECT id, state, created_at FROM states ORDER BY id").all()).toEqual([
		{ id: STATE_ID, state: STATE_JSON, created_at: 1_700_000_000 },
	]);
	expect(
		db
			.query(
				"SELECT id, state, model, name, type, instruction, criteria, answer, response_model, created_at FROM oracle ORDER BY id",
			)
			.all(),
	).toEqual([
		...LEGACY_ORACLE.map(row => ({ ...row, response_model: null })),
		...(hasFreshRecord
			? [
					{
						id: 12,
						state: STATE_ID,
						model: MODEL,
						name: "fresh",
						type: "noul",
						instruction: "Is this newly evaluated?",
						criteria: "null",
						answer: '{"type":"noul","noul":0.625}',
						response_model: "jev-raw-origin-v2",
						created_at: expect.any(Number),
					},
				]
			: []),
	]);
	expect(
		db
			.query(
				"SELECT id, state, results, provider, model, input_tokens, output_tokens, price, created_at FROM usage ORDER BY id",
			)
			.all(),
	).toEqual([
		...LEGACY_USAGE,
		...(hasFreshRecord
			? [
					{
						id: 18,
						state: STATE_ID,
						results: "[12]",
						provider: "typesafe",
						model: "new-compatibility-label",
						input_tokens: 37,
						output_tokens: 5,
						price: 0.0625,
						created_at: expect.any(Number),
					},
				]
			: []),
	]);
}

function recordAndReopen(dbPath: string): void {
	const cache = JudgmentCache.open(dbPath);
	try {
		expectLegacyLookup(cache);
		const miss = cache.lookup(MODEL, FRESH_REQUEST);
		expect(miss.answers).toEqual({});
		cache.record(MODEL, miss, FRESH_REQUEST.questions, FRESH_RESULT);
	} finally {
		cache.close();
	}
	const reopened = JudgmentCache.open(dbPath);
	try {
		expectLegacyLookup(reopened);
		const fresh = reopened.lookup(MODEL, FRESH_REQUEST);
		expect(fresh.answers).toEqual(FRESH_RESULT.answers);
		expect(fresh.responseModel).toBe("jev-raw-origin-v2");
		const mixed = reopened.lookup(MODEL, {
			state: STATE,
			questions: { ...LEGACY_REQUEST.questions, ...FRESH_REQUEST.questions },
		});
		expect(mixed.answers).toEqual({ ...LEGACY_ANSWERS, ...FRESH_RESULT.answers });
		expect(mixed.responseModel).toBeUndefined();
	} finally {
		reopened.close();
	}
	expectPersistedRows(dbPath, true);
}

const CHILD_SOURCE = `
import assert from "node:assert/strict";
import { JudgmentCache } from "@oh-my-pi/pi-coding-agent/judgment/cache";
const [dbPath, fixtureJson] = process.argv.slice(1);
const fixture = JSON.parse(fixtureJson);
await Bun.write(Bun.stdout, "READY\\n");
assert.equal(await Bun.stdin.text(), "release\\n");
const cache = JudgmentCache.open(dbPath);
try {
	const hit = cache.lookup(fixture.model, fixture.request);
	assert.equal(hit.stateId, fixture.stateId);
	assert.equal(hit.state, fixture.state);
	assert.deepEqual(hit.answers, fixture.answers);
	assert.equal(hit.responseModel, undefined);
} finally {
	cache.close();
}
await Bun.write(Bun.stdout, "CLOSED\\n");
`;

type ChildProcess = Bun.Subprocess<"pipe", "pipe", "pipe">;
interface OpeningChild {
	process: ChildProcess;
	ready: Promise<string>;
	stdout: Promise<string>;
	stderr: Promise<string>;
}

function startOpeningChild(dbPath: string): OpeningChild {
	const fixture = JSON.stringify({
		model: MODEL,
		request: LEGACY_REQUEST,
		stateId: STATE_ID,
		state: STATE_JSON,
		answers: LEGACY_ANSWERS,
	});
	const child = Bun.spawn([process.execPath, "-e", CHILD_SOURCE, dbPath, fixture], {
		cwd: path.resolve(import.meta.dir, "../../.."),
		stdin: "pipe",
		stdout: "pipe",
		stderr: "pipe",
	});
	const ready = Promise.withResolvers<string>();
	// Drain while waiting so diagnostics cannot fill a pipe and deadlock the barrier.
	const stderr = new Response(child.stderr).text().catch(error => `stderr read failed: ${String(error)}`);
	const stdout = (async () => {
		const reader = child.stdout.getReader();
		const decoder = new TextDecoder();
		let output = "";
		try {
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				output += decoder.decode(value, { stream: true });
				const newline = output.indexOf("\n");
				if (newline !== -1) ready.resolve(output.slice(0, newline));
			}
			return output + decoder.decode();
		} catch (error) {
			return `${output}\nstdout read failed: ${String(error)}`;
		} finally {
			ready.resolve("EOF before READY");
			reader.releaseLock();
		}
	})();
	return { process: child, ready: ready.promise, stdout, stderr };
}

async function contendOpeningLegacyDatabase(dbPath: string): Promise<void> {
	const children: OpeningChild[] = [];
	const killChildren = () => {
		for (const child of children) {
			if (child.process.exitCode !== null) continue;
			try {
				child.process.kill("SIGKILL");
			} catch {
				// The process can exit between the exitCode check and the signal.
			}
		}
	};
	let phase = "readiness barrier";
	let timedOut: string | undefined;
	// Real process I/O needs real timers. Bound both startup readiness and total lifetime;
	// a short watchdog can kill healthy Bun transpilation on a loaded CI runner.
	const watchdog = setTimeout(() => {
		timedOut = phase;
		killChildren();
	}, 20_000);
	let failed = false;
	let failure: unknown;
	try {
		for (let i = 0; i < 4; i++) children.push(startOpeningChild(dbPath));
		expect(await Promise.all(children.map(child => child.ready))).toEqual(["READY", "READY", "READY", "READY"]);
		phase = "concurrent open/close";
		// No child can open until every independent process has reached READY.
		await Promise.all(children.map(child => child.process.stdin.write("release\n")));
		await Promise.all(children.map(child => child.process.stdin.end()));
		await Promise.all(children.map(child => child.process.exited));
	} catch (error) {
		failed = true;
		failure = error;
	} finally {
		killChildren();
		await Promise.allSettled(children.map(child => child.process.exited));
		clearTimeout(watchdog);
	}
	const reports = await Promise.all(
		children.map(async (child, index) => ({
			index,
			exitCode: child.process.exitCode,
			signalCode: child.process.signalCode,
			stdout: await child.stdout,
			stderr: await child.stderr,
		})),
	);
	if (failed || timedOut || reports.some(report => report.exitCode !== 0 || report.stdout !== "READY\nCLOSED\n")) {
		const reason = timedOut ? `20s watchdog expired during ${timedOut}` : failed ? String(failure) : "Child failed";
		throw new Error(`${reason}\n${JSON.stringify(reports, null, 2)}`, { cause: failure });
	}
}

describe("JudgmentCache legacy migration", () => {
	it("preserves legacy answers and accounting with unknown origins across migration and reopen", () => {
		using tempDir = TempDir.createSync("@omp-judgment-migration-");
		const dbPath = path.join(tempDir.path(), "judgment-cache.db");
		seedLegacyDatabase(dbPath);
		for (let opening = 0; opening < 2; opening++) {
			const cache = JudgmentCache.open(dbPath);
			try {
				expectLegacyLookup(cache);
			} finally {
				cache.close();
			}
			expectPersistedRows(dbPath);
		}
		recordAndReopen(dbPath);
	});

	it("lets synchronized independent processes migrate one legacy DB without losing data or duplicating the column", async () => {
		using tempDir = TempDir.createSync("@omp-judgment-migration-processes-");
		const dbPath = path.join(tempDir.path(), "judgment-cache.db");
		seedLegacyDatabase(dbPath);
		await contendOpeningLegacyDatabase(dbPath);
		expectPersistedRows(dbPath);
		recordAndReopen(dbPath);
	}, 40_000);
});
