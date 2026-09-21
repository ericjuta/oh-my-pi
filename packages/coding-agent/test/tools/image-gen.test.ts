import { afterAll, describe, expect, it } from "bun:test";
import * as os from "node:os";
import * as path from "node:path";
import { type Api, type FetchImpl, type Model } from "@oh-my-pi/pi-ai";
import { ProviderHttpError } from "@oh-my-pi/pi-ai/error";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { modelKind } from "@oh-my-pi/pi-catalog/types";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import type { CustomToolContext, CustomToolResult } from "@oh-my-pi/pi-coding-agent/extensibility/custom-tools";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import {
	getImageGenTools,
	getImageGenToolsWithRegistry,
	imageGenTool,
} from "@oh-my-pi/pi-coding-agent/tools/image-gen";
import { removeWithRetries } from "@oh-my-pi/pi-utils";
import type { FormData } from "undici-types";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const generatedImagePaths: string[] = [];
const PNG_DATA = Buffer.from("catalog-image").toString("base64");
const WEBP_DATA = Buffer.from("hosted-image").toString("base64");

// Complete 1x1 PNG and GIF images, also used for path-based MIME detection.
const PNG_FIXTURE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
const GIF_FIXTURE_B64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==";

afterAll(async () => {
	await Promise.all(generatedImagePaths.map(imagePath => removeWithRetries(imagePath)));
});

function catalogModel(provider: string, id: string, api: Api, kind: "chat" | "image" = "image"): Model<Api> {
	return buildModel({
		id,
		name: `${provider}/${id}`,
		api,
		provider,
		baseUrl: `https://${provider}.example/v1`,
		kind,
		reasoning: false,
		input: kind === "image" ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 8_192,
	});
}

function createRegistry(models: Model<Api>[], credentials: Record<string, string | undefined> = {}): ModelRegistry {
	const authStorage = createInMemoryAuthStorage();
	for (const model of models) {
		const credential = credentials[model.provider];
		if (credential !== undefined) authStorage.setRuntimeApiKey(model.provider, credential);
		else if (!(model.provider in credentials)) authStorage.setRuntimeApiKey(model.provider, `key-${model.provider}`);
	}
	const registry = new ModelRegistry(authStorage);
	registry.getAvailable = (kind = "chat") =>
		kind === "all" ? [...models] : models.filter(model => modelKind(model) === kind);
	registry.getAll = () => [...models];
	registry.find = (provider, modelId) => models.find(model => model.provider === provider && model.id === modelId);
	return registry;
}

function createContext(options: {
	models: Model<Api>[];
	settings: Settings;
	fetch: FetchImpl;
	activeModel?: Model<Api>;
	credentials?: Record<string, string | undefined>;
	configuredHeaders?: Record<string, string>;
}): CustomToolContext {
	const modelRegistry = createRegistry(options.models, options.credentials);
	const { configuredHeaders } = options;
	// Reuse the record so edits cannot silently mutate later generation headers.
	if (configuredHeaders) modelRegistry.resolveModelHeaders = async () => configuredHeaders;
	return {
		fetch: options.fetch,
		sessionManager: SessionManager.inMemory("/tmp"),
		modelRegistry,
		model: options.activeModel,
		settings: options.settings,
		isIdle: () => true,
		hasQueuedMessages: () => false,
		abort: () => {},
	};
}

function imageResponse(mediaType = "image/png", data = PNG_DATA): Response {
	return new Response(JSON.stringify({ data: [{ b64_json: data, media_type: mediaType }] }), {
		status: 200,
		headers: { "content-type": "application/json" },
	});
}

function hostedResponse(): Response {
	return new Response(
		JSON.stringify({
			output: [
				{
					type: "image_generation_call",
					result: WEBP_DATA,
					revised_prompt: "catalog-hosted prompt",
					status: "completed",
				},
			],
		}),
		{ status: 200, headers: { "content-type": "application/json" } },
	);
}

function collectPaths(result: CustomToolResult<{ imagePaths: string[] }>): void {
	generatedImagePaths.push(...(result.details?.imagePaths ?? []));
}

type CapturedBody = { kind: "json"; value: unknown } | { kind: "multipart"; form: FormData };

interface CapturedRequest {
	url: string;
	method: string;
	headers: Headers;
	body: CapturedBody;
}

function recordRequests(respond: () => Response = () => imageResponse("image/png", PNG_FIXTURE_B64)): {
	fetch: FetchImpl;
	requests: CapturedRequest[];
} {
	const requests: CapturedRequest[] = [];
	const fetchImpl: FetchImpl = async (input, init) => {
		// Serialize and parse the wire body, not the original FormData object.
		const request = input instanceof Request ? new Request(input, init) : new Request(input.toString(), init);
		const body: CapturedBody = request.headers.get("content-type")?.startsWith("multipart/form-data")
			? { kind: "multipart", form: await request.formData() }
			: { kind: "json", value: await request.json() };
		requests.push({
			url: request.url,
			method: request.method,
			headers: request.headers,
			body,
		});
		return respond();
	};
	return { fetch: fetchImpl, requests };
}

function jsonBody(request: CapturedRequest | undefined): unknown {
	if (!request) throw new Error("Expected a recorded request");
	if (request.body.kind !== "json") throw new Error(`Expected a JSON request body, got ${request.body.kind}`);
	return request.body.value;
}

function multipartBody(request: CapturedRequest | undefined): FormData {
	if (!request) throw new Error("Expected a recorded request");
	if (request.body.kind !== "multipart")
		throw new Error(`Expected a multipart request body, got ${request.body.kind}`);
	return request.body.form;
}

async function formFiles(form: FormData, key: string): Promise<Array<{ type: string; base64: string }>> {
	const files: Array<{ type: string; base64: string }> = [];
	for (const entry of form.getAll(key)) {
		if (!(entry instanceof File)) throw new Error(`Expected ${key} entries to be files, got ${typeof entry}`);
		files.push({
			type: entry.type,
			base64: Buffer.from(await entry.arrayBuffer()).toString("base64"),
		});
	}
	return files;
}

describe("imageGenTool catalog routing", () => {
	it("registers without resolving credentials", async () => {
		const registry = createRegistry([]);
		expect(await getImageGenTools(registry)).toEqual([imageGenTool]);
		expect(await getImageGenToolsWithRegistry(registry)).toEqual([imageGenTool]);
		const schema = JSON.stringify(imageGenTool.parameters.toJsonSchema());
		expect(schema).toContain('"model"');
		expect(schema).not.toContain('"provider"');
	});

	it("uses a request model override as a single explicit candidate", async () => {
		const configured = catalogModel("deepinfra", "configured-image", "openai-images");
		const requested = catalogModel("openrouter", "requested-image", "openrouter-images");
		const urls: string[] = [];
		const fetchMock: FetchImpl = async input => {
			urls.push(input.toString());
			return imageResponse();
		};
		const settings = Settings.isolated({
			modelRoles: { image: "deepinfra/configured-image" },
			"retry.fallbackChains": { image: ["openrouter/requested-image"] },
		});
		const ctx = createContext({ models: [configured, requested], settings, fetch: fetchMock });

		const result = await imageGenTool.execute(
			"override",
			{ subject: "a catalog override", model: "openrouter/requested-image" },
			undefined,
			ctx,
		);
		collectPaths(result);

		expect(urls).toEqual(["https://openrouter.example/v1/images"]);
		expect(result.details?.model).toBe("requested-image");
	});

	it("hoists the active provider only within the non-explicit default image chain", async () => {
		const openai = catalogModel("openai", "gpt-image-1", "openai-responses");
		const xai = catalogModel("xai", "grok-imagine-image", "openai-images");
		const active = catalogModel("xai", "grok-chat", "openai-responses", "chat");
		const urls: string[] = [];
		const fetchMock: FetchImpl = async input => {
			urls.push(input.toString());
			return imageResponse();
		};
		const ctx = createContext({
			models: [openai, xai, active],
			settings: Settings.isolated(),
			fetch: fetchMock,
			activeModel: active,
		});

		const result = await imageGenTool.execute("hoist", { subject: "active provider" }, undefined, ctx);
		collectPaths(result);

		expect(urls).toEqual(["https://xai.example/v1/images/generations"]);
		expect(result.details?.provider).toBe("xai");
	});

	it("skips a candidate whose configured authentication is unavailable", async () => {
		const unavailable = catalogModel("deepinfra", "first-image", "openai-images");
		const fallback = catalogModel("openrouter", "fallback-image", "openrouter-images");
		const urls: string[] = [];
		const fetchMock: FetchImpl = async input => {
			urls.push(input.toString());
			return imageResponse();
		};
		const settings = Settings.isolated({
			modelRoles: { image: "deepinfra/first-image" },
			"retry.fallbackChains": { image: ["openrouter/fallback-image"] },
		});
		const ctx = createContext({
			models: [unavailable, fallback],
			settings,
			fetch: fetchMock,
			credentials: { deepinfra: undefined },
		});

		const result = await imageGenTool.execute("auth-skip", { subject: "fallback" }, undefined, ctx);
		collectPaths(result);

		expect(urls).toEqual(["https://openrouter.example/v1/images"]);
		expect(result.details?.model).toBe("fallback-image");
	});

	it("skips unsupported image APIs and continues the resolved chain", async () => {
		const unsupported = catalogModel("custom", "unsupported-image", "openai-completions");
		const fallback = catalogModel("openrouter", "fallback-image", "openrouter-images");
		const settings = Settings.isolated({
			modelRoles: { image: "custom/unsupported-image" },
			"retry.fallbackChains": { image: ["openrouter/fallback-image"] },
		});
		const urls: string[] = [];
		const fetchMock: FetchImpl = async input => {
			urls.push(input.toString());
			return imageResponse();
		};
		const ctx = createContext({ models: [unsupported, fallback], settings, fetch: fetchMock });

		const result = await imageGenTool.execute("unsupported", { subject: "fallback" }, undefined, ctx);
		collectPaths(result);

		expect(urls).toEqual(["https://openrouter.example/v1/images"]);
		expect(result.details?.model).toBe("fallback-image");
	});

	it("advances after provider HTTP failures and aggregates an exhausted chain", async () => {
		const first = catalogModel("deepinfra", "first-image", "openai-images");
		const second = catalogModel("openrouter", "second-image", "openrouter-images");
		const settings = Settings.isolated({
			modelRoles: { image: "deepinfra/first-image" },
			"retry.fallbackChains": { image: ["openrouter/second-image"] },
		});
		let calls = 0;
		const succeedingFetch: FetchImpl = async () => {
			calls++;
			if (calls === 1) return new Response(JSON.stringify({ error: { message: "first failed" } }), { status: 503 });
			return imageResponse();
		};
		const successContext = createContext({ models: [first, second], settings, fetch: succeedingFetch });
		const result = await imageGenTool.execute("http-fallback", { subject: "fallback" }, undefined, successContext);
		collectPaths(result);
		expect(result.details?.model).toBe("second-image");

		const failingFetch: FetchImpl = async () =>
			new Response(JSON.stringify({ error: { message: "failed" } }), { status: 503 });
		const failingContext = createContext({ models: [first, second], settings, fetch: failingFetch });
		await expect(
			imageGenTool.execute("aggregate", { subject: "fails" }, undefined, failingContext),
		).rejects.toBeInstanceOf(AggregateError);
	});

	it("propagates transport I/O failures without trying the next model", async () => {
		const first = catalogModel("deepinfra", "first-image", "openai-images");
		const second = catalogModel("openrouter", "second-image", "openrouter-images");
		const settings = Settings.isolated({
			modelRoles: { image: "deepinfra/first-image" },
			"retry.fallbackChains": { image: ["openrouter/second-image"] },
		});
		let calls = 0;
		const fetchMock: FetchImpl = async () => {
			calls++;
			throw new TypeError("socket closed");
		};
		const ctx = createContext({ models: [first, second], settings, fetch: fetchMock });

		await expect(imageGenTool.execute("io", { subject: "network" }, undefined, ctx)).rejects.toThrow("socket closed");
		expect(calls).toBe(1);
	});

	it("aggregates a standard edit 404 without retrying as generation or inventing a size", async () => {
		const model = catalogModel("deepinfra", "flux-selected", "openai-images");
		const transport = recordRequests(
			() =>
				new Response(JSON.stringify({ error: { message: "missing edits" } }), {
					status: 404,
					headers: { "content-type": "application/json" },
				}),
		);
		const settings = Settings.isolated({ modelRoles: { image: "deepinfra/flux-selected" } });
		const ctx = createContext({ models: [model], settings, fetch: transport.fetch });
		let failure: unknown;
		try {
			const result = await imageGenTool.execute(
				"edit-404",
				{ subject: "edit reference", input: [{ data: `data:image/png;base64,${PNG_FIXTURE_B64}` }] },
				undefined,
				ctx,
			);
			collectPaths(result);
		} catch (error) {
			failure = error;
		}

		expect(failure).toBeInstanceOf(AggregateError);
		if (!(failure instanceof AggregateError)) throw new Error("Expected an exhausted image chain");
		const errors: readonly unknown[] = failure.errors;
		expect(errors).toHaveLength(1);
		expect(errors[0]).toBeInstanceOf(ProviderHttpError);
		expect(errors[0]).toMatchObject({ status: 404, message: expect.stringContaining("missing edits") });
		expect(transport.requests.map(request => request.url)).toEqual(["https://deepinfra.example/v1/images/edits"]);
		const request = transport.requests[0];
		expect(request?.method).toBe("POST");
		expect(request?.headers.get("content-type")).toMatch(/^multipart\/form-data;\s*boundary=.+$/);
		const form = multipartBody(request);
		expect([...form.keys()].sort()).toEqual(["image[]", "model", "n", "prompt"]);
		expect(form.get("model")).toBe("flux-selected");
		expect(form.get("prompt")).toContain("edit reference");
		expect(form.get("n")).toBe("1");
		expect(form.get("size")).toBeNull();
		expect(await formFiles(form, "image[]")).toEqual([{ type: "image/png", base64: PNG_FIXTURE_B64 }]);
	});

	it("keeps all custom-provider multipart references and headers intact for subsequent JSON generation", async () => {
		const model = catalogModel("custom-lb", "selected-image", "openai-images");
		const bearer = "synthetic-custom-lb-image-token";
		const configuredHeaders = {
			"cOnTeNt-TyPe": "application/json; charset=utf-8",
			"X-LB-Route": "image-pool",
			"X-Request-Tag": "multipart-regression",
			Authorization: "Bearer ignored-configured-image-token",
		};
		const originalHeaders = { ...configuredHeaders };
		const transport = recordRequests();
		const ctx = createContext({
			models: [model],
			settings: Settings.isolated({ modelRoles: { image: "custom-lb/selected-image" } }),
			fetch: transport.fetch,
			credentials: { "custom-lb": bearer },
			configuredHeaders,
		});
		const inputPath = path.join(os.tmpdir(), `omp-image-gen-input-${crypto.randomUUID()}.gif`);
		try {
			await Bun.write(inputPath, Buffer.from(GIF_FIXTURE_B64, "base64"));
			const edited = await imageGenTool.execute(
				"custom-edit",
				{
					subject: "combine all references",
					image_size: "1024x1536",
					input: [
						{ path: inputPath },
						{ data: `data:image/png;base64,${PNG_FIXTURE_B64}` },
						{ data: GIF_FIXTURE_B64, mime_type: "image/gif" },
						{ data: PNG_FIXTURE_B64, mime_type: "image/png" },
					],
				},
				undefined,
				ctx,
			);
			collectPaths(edited);
			expect(configuredHeaders).toEqual(originalHeaders);
			const generated = await imageGenTool.execute(
				"custom-generation",
				{ subject: "generate without references" },
				undefined,
				ctx,
			);
			collectPaths(generated);

			expect(transport.requests.map(request => request.url)).toEqual([
				"https://custom-lb.example/v1/images/edits",
				"https://custom-lb.example/v1/images/generations",
			]);
			const [edit, generation] = transport.requests;
			expect(edit?.headers.get("content-type")).toMatch(/^multipart\/form-data;\s*boundary=.+$/);
			const form = multipartBody(edit);
			expect([...form.keys()].sort()).toEqual([
				"image[]",
				"image[]",
				"image[]",
				"image[]",
				"model",
				"n",
				"prompt",
				"size",
			]);
			expect(form.get("model")).toBe("selected-image");
			expect(form.get("prompt")).toContain("combine all references");
			expect(form.get("n")).toBe("1");
			expect(form.get("size")).toBe("1024x1536");
			expect(await formFiles(form, "image[]")).toEqual([
				{ type: "image/gif", base64: GIF_FIXTURE_B64 },
				{ type: "image/png", base64: PNG_FIXTURE_B64 },
				{ type: "image/gif", base64: GIF_FIXTURE_B64 },
				{ type: "image/png", base64: PNG_FIXTURE_B64 },
			]);
			expect(jsonBody(generation)).toEqual({
				model: "selected-image",
				prompt: expect.stringContaining("generate without references"),
				n: 1,
				response_format: "b64_json",
			});
			expect(generation?.headers.get("content-type")).toBe("application/json");
			for (const request of transport.requests) {
				expect(request.method).toBe("POST");
				expect(request.headers.get("authorization")).toBe(`Bearer ${bearer}`);
				expect(request.headers.get("x-lb-route")).toBe("image-pool");
				expect(request.headers.get("x-request-tag")).toBe("multipart-regression");
			}
			expect(configuredHeaders).toEqual(originalHeaders);
		} finally {
			await removeWithRetries(inputPath);
		}
	});

	it("keeps a single xai edit reference in the JSON image field", async () => {
		const model = catalogModel("xai", "grok-imagine-image", "openai-images");
		const bearer = "synthetic-xai-image-token";
		const transport = recordRequests();
		const ctx = createContext({
			models: [model],
			settings: Settings.isolated({ modelRoles: { image: "xai/grok-imagine-image" } }),
			fetch: transport.fetch,
			credentials: { xai: bearer },
			configuredHeaders: { "Content-Type": "application/json", "X-Image-Route": "xai-direct" },
		});
		const result = await imageGenTool.execute(
			"xai-edit",
			{ subject: "single reference", input: [{ data: PNG_FIXTURE_B64, mime_type: "image/png" }] },
			undefined,
			ctx,
		);
		collectPaths(result);

		expect(transport.requests.map(request => request.url)).toEqual(["https://xai.example/v1/images/edits"]);
		const request = transport.requests[0];
		expect(request?.method).toBe("POST");
		expect(request?.headers.get("content-type")).toBe("application/json");
		expect(request?.headers.get("authorization")).toBe(`Bearer ${bearer}`);
		expect(request?.headers.get("x-image-route")).toBe("xai-direct");
		const body = jsonBody(request);
		expect(body).toEqual({
			model: "grok-imagine-image",
			prompt: expect.stringContaining("single reference"),
			aspect_ratio: "1:1",
			resolution: "1k",
			n: 1,
			response_format: "b64_json",
			image: { type: "image_url", url: `data:image/png;base64,${PNG_FIXTURE_B64}` },
		});
		expect(body).not.toHaveProperty("input_references");
	});

	it("keeps xai-oauth edit references ordered in the JSON images array", async () => {
		const model = catalogModel("xai-oauth", "grok-imagine-image", "openai-images");
		const bearer = "synthetic-xai-oauth-image-token";
		const transport = recordRequests();
		const ctx = createContext({
			models: [model],
			settings: Settings.isolated({ modelRoles: { image: "xai-oauth/grok-imagine-image" } }),
			fetch: transport.fetch,
			credentials: { "xai-oauth": bearer },
			configuredHeaders: { "Content-Type": "application/json", "X-Image-Route": "xai-oauth" },
		});
		const result = await imageGenTool.execute(
			"xai-oauth-edit",
			{
				subject: "multiple references",
				aspect_ratio: "16:9",
				image_size: "1536x1024",
				input: [
					{ data: `data:image/gif;base64,${GIF_FIXTURE_B64}` },
					{ data: PNG_FIXTURE_B64, mime_type: "image/png" },
					{ data: GIF_FIXTURE_B64, mime_type: "image/gif" },
				],
			},
			undefined,
			ctx,
		);
		collectPaths(result);

		expect(transport.requests.map(request => request.url)).toEqual(["https://xai-oauth.example/v1/images/edits"]);
		const request = transport.requests[0];
		expect(request?.method).toBe("POST");
		expect(request?.headers.get("content-type")).toBe("application/json");
		expect(request?.headers.get("authorization")).toBe(`Bearer ${bearer}`);
		expect(request?.headers.get("x-image-route")).toBe("xai-oauth");
		const body = jsonBody(request);
		expect(body).toEqual({
			model: "grok-imagine-image",
			prompt: expect.stringContaining("multiple references"),
			aspect_ratio: "16:9",
			resolution: "2k",
			n: 1,
			response_format: "b64_json",
			images: [
				{ type: "image_url", url: `data:image/gif;base64,${GIF_FIXTURE_B64}` },
				{ type: "image_url", url: `data:image/png;base64,${PNG_FIXTURE_B64}` },
				{ type: "image_url", url: `data:image/gif;base64,${GIF_FIXTURE_B64}` },
			],
		});
		expect(body).not.toHaveProperty("input_references");
	});

	it("uses OpenRouter native images with edit references and response media types", async () => {
		const model = catalogModel("openrouter", "native-image", "openrouter-images");
		let requestBody: Record<string, unknown> | undefined;
		const fetchMock: FetchImpl = async (_input, init) => {
			requestBody = JSON.parse(String(init?.body));
			return imageResponse("image/webp");
		};
		const settings = Settings.isolated({ modelRoles: { image: "openrouter/native-image" } });
		const ctx = createContext({ models: [model], settings, fetch: fetchMock });

		const result = await imageGenTool.execute(
			"openrouter",
			{
				subject: "native edit",
				aspect_ratio: "16:9",
				input: [{ data: PNG_DATA, mime_type: "image/png" }],
			},
			undefined,
			ctx,
		);
		collectPaths(result);

		expect(requestBody).toMatchObject({
			model: "native-image",
			n: 1,
			aspect_ratio: "16:9",
			input_references: [{ type: "image_url", image_url: { url: `data:image/png;base64,${PNG_DATA}` } }],
		});
		expect(result.details?.images[0]?.mimeType).toBe("image/webp");
	});

	it("uses the selected Gemini catalog id and base URL", async () => {
		const model = catalogModel("google", "gemini-selected-image", "google-generative-ai");
		let requestUrl: string | undefined;
		const fetchMock: FetchImpl = async input => {
			requestUrl = input.toString();
			return new Response(
				JSON.stringify({
					candidates: [{ content: { parts: [{ inlineData: { data: PNG_DATA, mimeType: "image/png" } }] } }],
				}),
				{ status: 200 },
			);
		};
		const settings = Settings.isolated({ modelRoles: { image: "google/gemini-selected-image" } });
		const ctx = createContext({ models: [model], settings, fetch: fetchMock });

		const result = await imageGenTool.execute("gemini", { subject: "gemini" }, undefined, ctx);
		collectPaths(result);

		expect(requestUrl).toBe("https://google.example/v1/models/gemini-selected-image:generateContent");
		expect(result.details?.model).toBe("gemini-selected-image");
	});

	it("falls back to the selected Antigravity id when discovery advertises no image model", async () => {
		const model = catalogModel("google-antigravity", "gemini-selected-image", "google-gemini-cli");
		const requestBodies: Array<Record<string, unknown>> = [];
		const fetchMock: FetchImpl = async (input, init) => {
			const url = input.toString();
			if (url.endsWith("/v1internal:fetchAvailableModels")) {
				return new Response(JSON.stringify({ imageGenerationModelIds: [] }), { status: 200 });
			}
			requestBodies.push(JSON.parse(String(init?.body)));
			return new Response(
				`data: ${JSON.stringify({ response: { candidates: [{ content: { role: "model", parts: [{ inlineData: { data: PNG_DATA, mimeType: "image/png" } }] } }] } })}\n\n`,
				{ status: 200, headers: { "content-type": "text/event-stream" } },
			);
		};
		const credentials = JSON.stringify({ token: "antigravity-token", projectId: "catalog-project" });
		const settings = Settings.isolated({ modelRoles: { image: "google-antigravity/gemini-selected-image" } });
		const ctx = createContext({
			models: [model],
			settings,
			fetch: fetchMock,
			credentials: { "google-antigravity": credentials },
		});

		const result = await imageGenTool.execute("antigravity", { subject: "fallback id" }, undefined, ctx);
		collectPaths(result);

		expect(requestBodies[0]).toMatchObject({ model: "gemini-selected-image", project: "catalog-project" });
		expect(result.details?.model).toBe("gemini-selected-image");
	});

	it("uses a same-provider hosted chat carrier and places the selected OpenAI image id on the tool", async () => {
		const image = catalogModel("openai", "gpt-image-selected", "openai-responses");
		const activeCarrier = catalogModel("openai", "gpt-5.5", "openai-responses", "chat");
		let requestBody: Record<string, unknown> | undefined;
		const fetchMock: FetchImpl = async (_input, init) => {
			requestBody = JSON.parse(String(init?.body));
			return hostedResponse();
		};
		const settings = Settings.isolated({ modelRoles: { image: "openai/gpt-image-selected" } });
		const ctx = createContext({
			models: [image, activeCarrier],
			settings,
			fetch: fetchMock,
			activeModel: activeCarrier,
		});

		const result = await imageGenTool.execute("hosted", { subject: "hosted" }, undefined, ctx);
		collectPaths(result);

		expect(requestBody).toMatchObject({
			model: "gpt-5.5",
			tools: [{ type: "image_generation", model: "gpt-image-selected" }],
		});
		expect(result.details?.model).toBe("gpt-image-selected");
	});

	it("omits the image tool model for Codex hosted image requests", async () => {
		const image = catalogModel("openai-codex", "gpt-image-selected", "openai-codex-responses");
		const carrier = catalogModel("openai-codex", "gpt-5.5", "openai-codex-responses", "chat");
		let requestBody: Record<string, unknown> | undefined;
		const fetchMock: FetchImpl = async (_input, init) => {
			requestBody = JSON.parse(String(init?.body));
			const event = {
				type: "response.completed",
				response: { output: [{ type: "image_generation_call", result: WEBP_DATA }] },
			};
			return new Response(`data: ${JSON.stringify(event)}\n\n`, {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			});
		};
		const settings = Settings.isolated({ modelRoles: { image: "openai-codex/gpt-image-selected" } });
		const ctx = createContext({ models: [image, carrier], settings, fetch: fetchMock });

		const result = await imageGenTool.execute("codex-hosted", { subject: "codex" }, undefined, ctx);
		collectPaths(result);

		expect(requestBody?.model).toBe("gpt-5.5");
		const tools = requestBody?.tools;
		if (!Array.isArray(tools)) throw new Error("Expected hosted image tools");
		expect(tools[0]).toMatchObject({ type: "image_generation" });
		expect(tools[0]).not.toHaveProperty("model");
	});
});
