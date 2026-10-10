import { describe, expect, it } from "bun:test";
import { Database } from "bun:sqlite";
import {
	type Answer,
	type JudgmentResult,
	type NoulAnswer,
	type NoulQuestion,
	type Questions,
	tokenUsage,
} from "@oh-my-pi/pi-ai";
import { type CachedAnswers, JudgmentCache } from "@oh-my-pi/pi-coding-agent/judgment/cache";
import { TempDir } from "@oh-my-pi/pi-utils";

/** Catalog route the cache is keyed by. */
const SELECTED = "openrouter/~typesafe/jev-latest";
/** `result.model` compatibility label: billed on usage rows, never a cache key or origin. */
const COMPAT = "jev-latest";
/** Raw provider response identity: answer origin, never a cache key. */
const RAW = "typesafe/jev-1.13.0";
const STATE = { file: "a.ts", body: "x" } as const;
/** Every question id shares one definition; the cache keys rows by id plus definition. */
const QUESTION: NoulQuestion = { type: "noul", instructions: "Is this relevant?" };

interface Billing {
	model?: string;
	responseModel?: string;
	input?: number;
	output?: number;
	price?: number;
}

interface UsageRow {
	results: string;
	provider: string;
	model: string;
	input_tokens: number;
	output_tokens: number;
	price: number;
}

interface OracleRow {
	id: number;
	model: string;
	name: string;
	answer: string;
	response_model: string | null;
}

function noul(value: number): NoulAnswer {
	return { type: "noul", noul: value };
}

/** Record one billed request for `answers`, reusing the state identity a lookup derives. */
function record(cache: JudgmentCache, answers: Record<string, Answer>, billing: Billing = {}): void {
	const asked: Questions = Object.fromEntries(Object.keys(answers).map(id => [id, QUESTION]));
	const result: JudgmentResult = {
		api: "typesafe",
		provider: "openrouter",
		model: billing.model ?? COMPAT,
		answers,
		usage: tokenUsage(billing.input ?? 10, billing.output ?? 0, billing.price ?? 0.01),
	};
	if (billing.responseModel !== undefined) result.responseModel = billing.responseModel;
	cache.record(SELECTED, cache.lookup(SELECTED, { state: STATE, questions: asked }), asked, result);
}

/** Consumer-visible lookup fields; `toStrictEqual` on this distinguishes an absent origin from `undefined`. */
function lookup(
	cache: JudgmentCache,
	ids: readonly string[],
	model = SELECTED,
): Omit<CachedAnswers, "stateId" | "state"> {
	const {
		stateId: _stateId,
		state: _state,
		...visible
	} = cache.lookup(model, {
		state: STATE,
		questions: Object.fromEntries(ids.map(id => [id, QUESTION])),
	});
	return visible;
}

function usageRows(db: Database): UsageRow[] {
	return db
		.query<UsageRow, []>("SELECT results, provider, model, input_tokens, output_tokens, price FROM usage ORDER BY id")
		.all();
}

function oracleRows(db: Database): OracleRow[] {
	return db.query<OracleRow, []>("SELECT id, model, name, answer, response_model FROM oracle ORDER BY id").all();
}

describe("JudgmentCache", () => {
	it("serves recorded answers with their raw origin across reopen, keyed only by the selected route", () => {
		using tempDir = TempDir.createSync("@omp-judgment-cache-identity-");
		const dbPath = tempDir.join("judgment-cache.db");
		let cache: JudgmentCache | undefined = JudgmentCache.open(dbPath);
		let inspect: Database | undefined;
		try {
			expect(lookup(cache, ["a", "b"])).toStrictEqual({ answers: {} });

			record(cache, { a: noul(0.9), b: noul(0.2) }, { responseModel: RAW, input: 12, output: 3, price: 0.024 });
			const warm = { answers: { a: noul(0.9), b: noul(0.2) }, responseModel: RAW };
			expect(lookup(cache, ["a", "b"])).toStrictEqual(warm);
			// Neither the raw origin nor the compatibility label may address cached answers.
			expect(lookup(cache, ["a", "b"], RAW)).toStrictEqual({ answers: {} });
			expect(lookup(cache, ["a", "b"], COMPAT)).toStrictEqual({ answers: {} });

			cache.close();
			cache = undefined;
			cache = JudgmentCache.open(dbPath);
			expect(lookup(cache, ["a", "b"])).toStrictEqual(warm);
			expect(lookup(cache, ["a", "b"], RAW)).toStrictEqual({ answers: {} });

			cache.close();
			cache = undefined;
			inspect = new Database(dbPath, { readonly: true });
			const oracle = oracleRows(inspect);
			expect(oracle).toEqual([
				{
					id: expect.any(Number),
					model: SELECTED,
					name: "a",
					answer: JSON.stringify(noul(0.9)),
					response_model: RAW,
				},
				{
					id: expect.any(Number),
					model: SELECTED,
					name: "b",
					answer: JSON.stringify(noul(0.2)),
					response_model: RAW,
				},
			]);
			// One billed request, stock usage fields; the five lookups above appended nothing.
			expect(usageRows(inspect)).toEqual([
				{
					results: JSON.stringify(oracle.map(row => row.id)),
					provider: "openrouter",
					model: COMPAT,
					input_tokens: 12,
					output_tokens: 3,
					price: 0.024,
				},
			]);
		} finally {
			cache?.close();
			inspect?.close();
		}
	});

	it("reports a raw origin only when every returned hit shares one known origin", () => {
		using tempDir = TempDir.createSync("@omp-judgment-cache-partial-");
		const dbPath = tempDir.join("judgment-cache.db");
		let cache: JudgmentCache | undefined = JudgmentCache.open(dbPath);
		let inspect: Database | undefined;
		try {
			record(cache, { same1: noul(0.11), same2: noul(0.12) }, { responseModel: "raw-a" });
			record(cache, { other: noul(0.2) }, { responseModel: "raw-b" });
			record(cache, { nullOrigin: noul(0.3) });
			record(cache, { emptyOrigin: noul(0.4) }, { responseModel: "" });

			expect(
				[
					["same1", "same2", "missing"],
					["same1", "other"],
					["same1", "nullOrigin"],
					["emptyOrigin"],
					["missing"],
				].map(ids => lookup(cache!, ids)),
			).toStrictEqual([
				// Missing questions and unrequested unknown/different-origin rows do not taint known hits.
				{ answers: { same1: noul(0.11), same2: noul(0.12) }, responseModel: "raw-a" },
				// Different known origins, or a known hit beside an unknown one, have no shared origin.
				{ answers: { same1: noul(0.11), other: noul(0.2) } },
				{ answers: { same1: noul(0.11), nullOrigin: noul(0.3) } },
				// An empty raw identity is unknown, not a known origin.
				{ answers: { emptyOrigin: noul(0.4) } },
				{ answers: {} },
			]);

			cache.close();
			cache = undefined;
			inspect = new Database(dbPath, { readonly: true });
			// An omitted result.responseModel persists as SQL NULL, not a string or the compatibility label.
			expect(oracleRows(inspect).map(({ name, response_model }) => ({ name, response_model }))).toEqual([
				{ name: "same1", response_model: "raw-a" },
				{ name: "same2", response_model: "raw-a" },
				{ name: "other", response_model: "raw-b" },
				{ name: "nullOrigin", response_model: null },
				{ name: "emptyOrigin", response_model: "" },
			]);
		} finally {
			cache?.close();
			inspect?.close();
		}
	});

	it("overwrites an answer and its origin in place, billing each successful record once", () => {
		using tempDir = TempDir.createSync("@omp-judgment-cache-overwrite-");
		const dbPath = tempDir.join("judgment-cache.db");
		let cache: JudgmentCache | undefined = JudgmentCache.open(dbPath);
		let inspect: Database | undefined;
		try {
			record(cache, { a: noul(0.1) }, { responseModel: "raw-a", input: 1, price: 0.001 });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.1) }, responseModel: "raw-a" });

			// Known origin → different known origin.
			record(cache, { a: noul(0.6) }, { responseModel: "raw-b", input: 2, price: 0.002 });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.6) }, responseModel: "raw-b" });

			// Known origin → unknown: the stale origin must not survive the new answer.
			record(cache, { a: noul(0.9) }, { input: 3, price: 0.003 });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.9) } });

			cache.close();
			cache = undefined;
			cache = JudgmentCache.open(dbPath);
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.9) } });

			cache.close();
			cache = undefined;
			inspect = new Database(dbPath, { readonly: true });
			const oracle = oracleRows(inspect);
			expect(oracle).toEqual([
				{
					id: expect.any(Number),
					model: SELECTED,
					name: "a",
					answer: JSON.stringify(noul(0.9)),
					response_model: null,
				},
			]);
			const results = JSON.stringify([oracle[0].id]);
			expect(usageRows(inspect)).toEqual(
				[1, 2, 3].map(input => ({
					results,
					provider: "openrouter",
					model: COMPAT,
					input_tokens: input,
					output_tokens: 0,
					price: input / 1000,
				})),
			);
		} finally {
			cache?.close();
			inspect?.close();
		}
	});

	it("invalidates raw origins on legacy answer writes, including same-value upserts", () => {
		using tempDir = TempDir.createSync("@omp-judgment-cache-legacy-writer-");
		const dbPath = tempDir.join("judgment-cache.db");
		let cache: JudgmentCache | undefined = JudgmentCache.open(dbPath);
		const peer = new Database(dbPath);
		try {
			record(cache, { a: noul(0.1) }, { responseModel: RAW, input: 1, price: 0.001 });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.1) }, responseModel: RAW });
			const id = oracleRows(peer)[0].id;
			const { stateId } = cache.lookup(SELECTED, { state: STATE, questions: { a: QUESTION } });
			// Exact legacy writer shape: eight columns, with no knowledge of response_model.
			const legacyUpsert = peer.prepare<
				{ id: number },
				[string, string, string, string, string, string, string, number]
			>(`
INSERT INTO oracle (state, model, name, type, instruction, criteria, answer, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (state, model, name, type, instruction, criteria)
DO UPDATE SET answer = excluded.answer, created_at = excluded.created_at
RETURNING id`);
			expect(
				legacyUpsert.get(
					stateId,
					SELECTED,
					"a",
					QUESTION.type,
					QUESTION.instructions,
					"null",
					JSON.stringify(noul(0.6)),
					1,
				),
			).toEqual({ id });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.6) } });
			expect(oracleRows(peer)[0].response_model).toBeNull();

			// Restore a known origin, then have the old writer publish the identical answer bytes.
			record(cache, { a: noul(0.6) }, { responseModel: RAW, input: 2, price: 0.002 });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.6) }, responseModel: RAW });
			expect(
				legacyUpsert.get(
					stateId,
					SELECTED,
					"a",
					QUESTION.type,
					QUESTION.instructions,
					"null",
					JSON.stringify(noul(0.6)),
					1,
				),
			).toEqual({ id });
			expect(lookup(cache, ["a"])).toStrictEqual({ answers: { a: noul(0.6) } });
			expect(oracleRows(peer)[0].response_model).toBeNull();

			// Current writers restore and retain the same known origin, even for an unchanged answer.
			record(cache, { a: noul(0.6) }, { responseModel: RAW, input: 3, price: 0.003 });
			const known = { answers: { a: noul(0.6) }, responseModel: RAW };
			expect(lookup(cache, ["a"])).toStrictEqual(known);
			record(cache, { a: noul(0.6) }, { responseModel: RAW, input: 4, price: 0.004 });
			expect(lookup(cache, ["a"])).toStrictEqual(known);

			cache.close();
			cache = undefined;
			cache = JudgmentCache.open(dbPath);
			expect(lookup(cache, ["a"])).toStrictEqual(known);
			expect(oracleRows(peer)).toEqual([
				{ id, model: SELECTED, name: "a", answer: JSON.stringify(noul(0.6)), response_model: RAW },
			]);
			// Each current record bills once; trigger/stamp writes, lookups, and reopen add no usage.
			expect(usageRows(peer)).toEqual(
				[1, 2, 3, 4].map(input => ({
					results: JSON.stringify([id]),
					provider: "openrouter",
					model: COMPAT,
					input_tokens: input,
					output_tokens: 0,
					price: input / 1000,
				})),
			);
		} finally {
			cache?.close();
			peer.close();
		}
	});

	it("rolls back a whole record when its usage insert aborts, keeping prior answers and origin", () => {
		using tempDir = TempDir.createSync("@omp-judgment-cache-abort-");
		const dbPath = tempDir.join("judgment-cache.db");
		let cache: JudgmentCache | undefined = JudgmentCache.open(dbPath);
		const peer = new Database(dbPath);
		try {
			record(cache, { a: noul(0.3) }, { responseModel: "raw-a" });
			peer.run(
				"CREATE TRIGGER reject_usage BEFORE INSERT ON usage BEGIN SELECT RAISE(ABORT, 'usage rejected'); END",
			);

			// Best-effort: the failed write is swallowed, not surfaced to the judgment.
			expect(() =>
				record(cache!, { a: noul(0.8), b: noul(0.5) }, { responseModel: "raw-b", input: 99, price: 0.099 }),
			).not.toThrow();
			const prior = { answers: { a: noul(0.3) }, responseModel: "raw-a" };
			expect(lookup(cache, ["a", "b"])).toStrictEqual(prior);

			cache.close();
			cache = undefined;
			cache = JudgmentCache.open(dbPath);
			expect(lookup(cache, ["a", "b"])).toStrictEqual(prior);

			cache.close();
			cache = undefined;
			const oracle = oracleRows(peer);
			expect(oracle).toEqual([
				{
					id: expect.any(Number),
					model: SELECTED,
					name: "a",
					answer: JSON.stringify(noul(0.3)),
					response_model: "raw-a",
				},
			]);
			expect(usageRows(peer)).toEqual([
				{
					results: JSON.stringify([oracle[0].id]),
					provider: "openrouter",
					model: COMPAT,
					input_tokens: 10,
					output_tokens: 0,
					price: 0.01,
				},
			]);
		} finally {
			cache?.close();
			if (peer.inTransaction) peer.run("ROLLBACK");
			peer.close();
		}
	});
});
