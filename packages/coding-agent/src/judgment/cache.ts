/**
 * Local answer cache for native System One judgments (TypeSafe jev, directly
 * or through OpenRouter).
 *
 * Every request decomposes into its parts: the state it evaluated, one oracle
 * row per question answered about that state, and one usage row per billed
 * provider request. Questions are answered independently of their siblings,
 * so a later request that re-asks any question about an identical state under
 * the same model is answered locally for free, and only its unanswered
 * questions reach the provider.
 *
 * ```text
 * states  (id = sha256(canonical state JSON), state)
 * oracle  (id, state → states.id, model, name, type, instruction, criteria, answer, response_model)
 *         UNIQUE (state, model, name, type, instruction, criteria)
 * usage   (id, state → states.id, results = JSON [oracle.id…], provider, model, tokens, price)
 * ```
 */
import type { Database, Statement } from "bun:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";
import type { Answer, JudgmentRequest, JudgmentResult, Question, Questions } from "@oh-my-pi/pi-ai";
import {
	checkpointWal,
	getJudgmentCacheDbPath,
	isBunTestRuntime,
	logger,
	openSqliteDatabaseSync,
	postmortem,
	stableStringifyJson,
} from "@oh-my-pi/pi-utils";

const SCHEMA = `
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS states (
	id TEXT PRIMARY KEY,
	state TEXT NOT NULL,
	created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS oracle (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	state TEXT NOT NULL REFERENCES states(id),
	model TEXT NOT NULL,
	name TEXT NOT NULL,
	type TEXT NOT NULL,
	instruction TEXT NOT NULL,
	criteria TEXT NOT NULL,
	answer TEXT NOT NULL,
	response_model TEXT,
	created_at INTEGER NOT NULL,
	UNIQUE (state, model, name, type, instruction, criteria)
);
CREATE TABLE IF NOT EXISTS usage (
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
CREATE INDEX IF NOT EXISTS idx_usage_state ON usage(state);
`;

/** Cached answers for one request, plus the state identity {@link JudgmentCache.record} reuses. */
export interface CachedAnswers {
	/** `sha256` of the canonical state JSON. */
	stateId: string;
	/** Canonical state JSON, stored once per distinct state. */
	state: string;
	/** Answers already known for the request's questions, keyed by question id. */
	answers: Record<string, Answer>;
	/** Original raw response model, only when all returned answers have the same known origin. */
	responseModel?: string;
}

interface QuestionKey {
	name: string;
	type: string;
	instruction: string;
	criteria: string;
}

function questionKey(name: string, question: Question): QuestionKey {
	return {
		name,
		type: question.type,
		instruction: question.instructions,
		// `null`, not SQL NULL: NULLs never collide under UNIQUE, which would let duplicates in.
		criteria: stableStringifyJson(question.criteria ?? null),
	};
}

function unixSeconds(): number {
	return Math.floor(Date.now() / 1000);
}

let shared: JudgmentCache | null | undefined;

/**
 * Process-wide cache at {@link getJudgmentCacheDbPath}, opened on first use
 * and closed at exit. `undefined` when the database cannot be opened (the
 * failure is logged once and judgments proceed uncached) and under the test
 * runner, where mocked answers must never persist into or be served from the
 * user's cache; tests inject {@link JudgmentCache.open} explicitly.
 */
export function sharedJudgmentCache(): JudgmentCache | undefined {
	if (isBunTestRuntime()) return undefined;
	if (shared === undefined) {
		try {
			const cache = JudgmentCache.open();
			postmortem.register("judgment-cache", () => cache.close(), { exitOnly: true });
			shared = cache;
		} catch (error) {
			logger.warn("judgment cache unavailable; judgments run uncached", { error: String(error) });
			shared = null;
		}
	}
	return shared ?? undefined;
}

/** SQLite store of native judgment answers and the billed requests that produced them. */
export class JudgmentCache {
	readonly #db: Database;
	readonly #answer: Statement<
		{ answer: string; response_model: string | null },
		[string, string, string, string, string, string]
	>;
	readonly #insertState: Statement<unknown, [string, string, number]>;
	readonly #upsertOracle: Statement<{ id: number }, [string, string, string, string, string, string, string, number]>;
	readonly #setResponseModel: Statement<unknown, [string | null, number]>;
	readonly #insertUsage: Statement<unknown, [string, string, string, string, number, number, number, number]>;

	private constructor(db: Database) {
		this.#db = db;
		db.run(SCHEMA);
		// Lock before inspecting the schema so concurrent openers cannot both add the column.
		// Legacy origins remain unknown: usage.model is not necessarily the raw response identity.
		db.transaction(() => {
			const columns = db.query<{ name: string }, []>("PRAGMA table_info(oracle)").all();
			if (!columns.some(column => column.name === "response_model")) {
				db.run("ALTER TABLE oracle ADD COLUMN response_model TEXT");
			}
			// Answer writes from older processes cannot retain an origin, even for an unchanged value.
			db.run(`
CREATE TRIGGER IF NOT EXISTS oracle_answer_invalidates_response_model
AFTER UPDATE OF answer ON oracle
WHEN NEW.response_model IS NOT NULL
BEGIN
	UPDATE oracle SET response_model = NULL WHERE id = NEW.id;
END`);
		}).immediate();
		this.#answer = db.prepare(
			"SELECT answer, response_model FROM oracle WHERE state = ? AND model = ? AND name = ? AND type = ? AND instruction = ? AND criteria = ?",
		);
		this.#insertState = db.prepare("INSERT OR IGNORE INTO states (id, state, created_at) VALUES (?, ?, ?)");
		this.#upsertOracle = db.prepare(`
INSERT INTO oracle (state, model, name, type, instruction, criteria, answer, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (state, model, name, type, instruction, criteria)
DO UPDATE SET answer = excluded.answer, created_at = excluded.created_at
RETURNING id`);
		this.#setResponseModel = db.prepare("UPDATE oracle SET response_model = ? WHERE id = ?");
		this.#insertUsage = db.prepare(`
INSERT INTO usage (state, results, provider, model, input_tokens, output_tokens, price, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)`);
	}

	/**
	 * Open (creating if needed) the cache at `dbPath`, quarantining a corrupt store once.
	 * @throws when the directory or database cannot be created.
	 */
	static open(dbPath: string = getJudgmentCacheDbPath()): JudgmentCache {
		fs.mkdirSync(path.dirname(dbPath), { recursive: true });
		return openSqliteDatabaseSync(dbPath, db => new JudgmentCache(db), { recoverCorruption: true });
	}

	/**
	 * Answers already recorded for `request`'s questions under `model`. Best-effort:
	 * a database failure is logged and reported as no cached answers.
	 */
	lookup(model: string, request: JudgmentRequest): CachedAnswers {
		const state = stableStringifyJson(request.state);
		const stateId = new Bun.CryptoHasher("sha256").update(state).digest("hex");
		const answers: Record<string, Answer> = {};
		let responseModel: string | undefined;
		let allOriginsKnown = true;
		try {
			for (const name in request.questions) {
				const key = questionKey(name, request.questions[name]);
				const row = this.#answer.get(stateId, model, key.name, key.type, key.instruction, key.criteria);
				if (row) {
					answers[name] = JSON.parse(row.answer);
					if (!row.response_model || (responseModel !== undefined && row.response_model !== responseModel)) {
						allOriginsKnown = false;
					} else {
						responseModel = row.response_model;
					}
				}
			}
		} catch (error) {
			logger.warn("judgment cache lookup failed", { error: String(error) });
			return { stateId, state, answers: {} };
		}
		const cached: CachedAnswers = { stateId, state, answers };
		if (allOriginsKnown && responseModel !== undefined) cached.responseModel = responseModel;
		return cached;
	}

	/**
	 * Record one billed request: its state, one oracle row per question it
	 * answered, and a usage row linking them to the price paid. Best-effort:
	 * failures are logged and never surface to the judgment.
	 */
	record(model: string, cached: CachedAnswers, questions: Questions, result: JudgmentResult): void {
		const now = unixSeconds();
		try {
			this.#db.transaction(() => {
				this.#insertState.run(cached.stateId, cached.state, now);
				const results: number[] = [];
				for (const name in questions) {
					const key = questionKey(name, questions[name]);
					const row = this.#upsertOracle.get(
						cached.stateId,
						model,
						key.name,
						key.type,
						key.instruction,
						key.criteria,
						JSON.stringify(result.answers[name]),
						now,
					);
					if (row) {
						// Stamp after answer-trigger invalidation, atomically with the answer and billing row.
						this.#setResponseModel.run(result.responseModel ?? null, row.id);
						results.push(row.id);
					}
				}
				this.#insertUsage.run(
					cached.stateId,
					JSON.stringify(results),
					result.provider,
					result.model,
					result.usage.input,
					result.usage.output,
					result.usage.cost.total,
					now,
				);
			})();
		} catch (error) {
			logger.warn("judgment cache write failed", { error: String(error) });
		}
	}

	/** Checkpoint and close the database; the cache must not be used afterwards. */
	close(): void {
		checkpointWal(this.#db);
		this.#db.close();
	}
}
