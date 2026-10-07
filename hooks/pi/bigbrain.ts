// bigbrain extension for the Pi coding agent.
//
// Pi has no MCP client, so this registers the bigbrain memory tools natively and bridges
// each call to the shared HTTP server. It also runs the post-turn memory-maintenance pass:
// on `agent_settled` it renders the turn that just ended and hands it to the standalone
// direct runner, detached, so the pass never appears in the session. The runner performs
// the pass with a headless `pi -p` that loads only this extension, so it uses the same
// providers and credentials as this Pi and needs no separate API key.
//
// Installed by `bigbrain install-hooks --target pi`, which also places the direct runner
// under ~/.pi/agent/bigbrain/ and stamps the repo path below. (Not ~/.pi/agent/hooks/: Pi
// warns about that directory as a leftover from before extensions.)
//
// Environment overrides:
//   BIGBRAIN_MCP_URL        bigbrain endpoint (default: http://127.0.0.1:8765/mcp)
//   BIGBRAIN_MAINT_RUNNER   explicit path to bigbrain-maintenance-direct.mjs
//   BIGBRAIN_MAINT_LOG      log file (default: ~/.bigbrain/maintenance.log)

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const MCP_URL = process.env.BIGBRAIN_URL || process.env.BIGBRAIN_MCP_URL || "http://127.0.0.1:8765/mcp";
const HOOK_DIR = join(tmpdir(), "bigbrain-hooks");
const LOG_FILE = process.env.BIGBRAIN_MAINT_LOG || join(homedir(), ".bigbrain", "maintenance.log");

// Budgets for the rendered window. User text gets far more room than assistant text: what
// the user said is the part most worth remembering, and it is usually short anyway.
function envInt(name: string, fallback: number): number {
	const n = Number.parseInt(process.env[name] ?? "", 10);
	return Number.isFinite(n) && n > 0 ? n : fallback;
}
const MAX_CHARS = envInt("BIGBRAIN_MAINT_MAX_CHARS", 60000);
const USER_CHARS = envInt("BIGBRAIN_MAINT_USER_CHARS", 16000);
const ASSISTANT_CHARS = 4000;

// The installer copies the runner next to this extension's config tree and stamps the
// checkout path as a fallback, so a fresh machine needs no hard-coded location.
const RUNNER_CANDIDATES = [
	process.env.BIGBRAIN_MAINT_RUNNER,
	join(homedir(), ".pi", "agent", "bigbrain", "bigbrain-maintenance-direct.mjs"),
	join(homedir(), ".pi", "agent", "hooks", "bigbrain-maintenance-direct.mjs"), // pre-migration installs
	join("__BIGBRAIN_REPO__", "hooks", "bigbrain-maintenance-direct.mjs"),
].filter((p): p is string => !!p);

function resolveRunner(): string | null {
	return RUNNER_CANDIDATES.find((p) => existsSync(p)) ?? null;
}

// The runner drives the pass through a headless Pi so it reuses this Pi's providers and
// credentials. Hand it the exact CLI entry point and this file, which the child loads as
// its only extension.
function thisExtensionPath(): string | undefined {
	try {
		return fileURLToPath(import.meta.url);
	} catch {
		return undefined;
	}
}

// A detached pass has nobody watching its stderr, so every skip must leave a log line.
function logNote(msg: string) {
	try {
		mkdirSync(dirname(LOG_FILE), { recursive: true });
		appendFileSync(LOG_FILE, `${new Date().toISOString()} ${msg}\n`);
	} catch {
		// Logging must never break the interactive session.
	}
}

async function callMcpTool(name: string, args: Record<string, unknown>, signal?: AbortSignal) {
	try {
		const res = await fetch(MCP_URL, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json, text/event-stream",
			},
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: Date.now(),
				method: "tools/call",
				params: {
					name,
					arguments: args,
				},
			}),
			signal,
		});

		if (!res.ok) {
			return {
				content: [{ type: "text" as const, text: `Bigbrain MCP HTTP error ${res.status}: ${res.statusText}` }],
				isError: true,
			};
		}

		const bodyText = await res.text();

		// Parse SSE response
		let resultJson: any = null;
		for (const line of bodyText.split("\n")) {
			if (line.startsWith("data: ")) {
				try {
					resultJson = JSON.parse(line.slice(6));
					break;
				} catch {
					// continue searching lines
				}
			}
		}

		if (!resultJson) {
			try {
				resultJson = JSON.parse(bodyText);
			} catch {
				return {
					content: [{ type: "text" as const, text: `Failed to parse bigbrain response:\n${bodyText}` }],
					isError: true,
				};
			}
		}

		if (resultJson.error) {
			return {
				content: [{ type: "text" as const, text: `Bigbrain error: ${resultJson.error.message || JSON.stringify(resultJson.error)}` }],
				isError: true,
			};
		}

		const result = resultJson.result;
		if (!result) {
			return {
				content: [{ type: "text" as const, text: JSON.stringify(resultJson, null, 2) }],
				isError: false,
			};
		}

		if (Array.isArray(result.content)) {
			return {
				content: result.content.map((c: any) => ({
					type: "text" as const,
					text: typeof c.text === "string" ? c.text : JSON.stringify(c, null, 2),
				})),
				isError: !!result.isError,
			};
		}

		return {
			content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
			isError: false,
		};
	} catch (err: any) {
		return {
			content: [{
				type: "text" as const,
				text: `Failed to connect to bigbrain MCP server at ${MCP_URL}: ${err.message || String(err)}.\nEnsure the server is running (e.g. launchctl start com.bigbrain.mcp or 'uv run bigbrain serve --port 8765').`,
			}],
			isError: true,
		};
	}
}

// The last session entry a pass has covered, per session, so the next pass starts right
// after it. Kept on disk so it survives /reload and resuming the session.
function checkpointFile(sessionId: string): string {
	return join(HOOK_DIR, `reviewed-pi-${sessionId.replace(/[^A-Za-z0-9._-]/g, "_")}`);
}

function readCheckpoint(sessionId: string): string | null {
	try {
		return readFileSync(checkpointFile(sessionId), "utf8").trim() || null;
	} catch {
		return null;
	}
}

function writeCheckpoint(sessionId: string, entryId: string) {
	try {
		mkdirSync(HOOK_DIR, { recursive: true });
		writeFileSync(checkpointFile(sessionId), entryId);
	} catch {
		// A lost checkpoint only means the next pass falls back to the last user prompt.
	}
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((c: any) => c?.type === "text" && typeof c.text === "string")
		.map((c: any) => c.text)
		.join("\n");
}

function render(msgs: any[], withTools: boolean): { text: string; hasTools: boolean } {
	let hasTools = false;
	const lines: string[] = [];

	for (const m of msgs) {
		if (m.role === "user") {
			const text = textOf(m.content);
			if (text.trim()) {
				lines.push(`USER: ${text.slice(0, USER_CHARS)}`);
			}
		} else if (m.role === "assistant") {
			if (Array.isArray(m.content)) {
				for (const block of m.content) {
					if (block.type === "text" && block.text?.trim()) {
						lines.push(`ASSISTANT: ${block.text.slice(0, ASSISTANT_CHARS)}`);
					} else if (block.type === "toolCall") {
						hasTools = true;
						const argsStr = withTools ? " " + JSON.stringify(block.arguments || {}).slice(0, 500) : "";
						lines.push(`TOOL ${block.name || "?"}${argsStr}`);
					}
				}
			} else if (typeof m.content === "string" && m.content.trim()) {
				lines.push(`ASSISTANT: ${m.content.slice(0, ASSISTANT_CHARS)}`);
			}
		} else if (m.role === "toolResult") {
			hasTools = true;
			const text = textOf(m.content);
			if (withTools && text.trim()) {
				lines.push(`RESULT ${text.slice(0, 600)}`);
			}
		} else if (m.role === "bashExecution") {
			hasTools = true;
			lines.push(`TOOL bash ${JSON.stringify({ command: m.command }).slice(0, 500)}`);
			if (withTools && m.output?.trim()) {
				lines.push(`RESULT ${m.output.slice(0, 600)}`);
			}
		}
	}

	return { text: lines.join("\n"), hasTools };
}

// Render everything since the previous pass: the whole run, including the original request
// and any message the user sent while the agent was working, plus any earlier turn that
// was skipped. With no usable checkpoint (first pass, or the checkpoint is on another
// /tree branch) it falls back to the last real user prompt, so a resumed session does not
// re-review its whole history.
function extractPendingTurns(ctx: ExtensionContext, sessionId: string): { turnText: string; hasTools: boolean; lastEntryId: string | null } {
	const entries = ctx.sessionManager.getBranch().filter((e: any) => e.type === "message" && e.message);
	if (entries.length === 0) {
		return { turnText: "", hasTools: false, lastEntryId: null };
	}
	const lastEntryId = entries[entries.length - 1].id;

	let start = -1;
	const checkpoint = readCheckpoint(sessionId);
	if (checkpoint) {
		const at = entries.findIndex((e: any) => e.id === checkpoint);
		if (at !== -1) start = at + 1;
	}
	if (start === -1) {
		start = 0;
		for (let i = entries.length - 1; i >= 0; i--) {
			const m = (entries[i] as any).message;
			if (m.role === "user" && textOf(m.content).trim()) {
				start = i;
				break;
			}
		}
	}

	const msgs = entries.slice(start).map((e: any) => e.message);
	let { text, hasTools } = render(msgs, true);
	// A wider window can blow the budget. Tool output is the bulk and the least durable
	// part, so shed it first; only then cut the middle, keeping the start of the window.
	if (text.length > MAX_CHARS) {
		text = render(msgs, false).text;
	}
	if (text.length > MAX_CHARS) {
		text = `${text.slice(0, 2000)}\n\n[... middle of turn omitted ...]\n\n${text.slice(-(MAX_CHARS - 2000))}`;
	}
	return { turnText: text, hasTools, lastEntryId };
}

function triggerDetachedMaintenance(ctx: ExtensionContext) {
	// Recursion guard
	if (process.env.BIGBRAIN_MAINT === "1") {
		return;
	}

	const sessionId = ctx.sessionManager.getSessionId() || "pi-session";
	const { turnText, hasTools, lastEntryId } = extractPendingTurns(ctx, sessionId);
	// Only run maintenance for substantive turns (at least one tool executed). A skipped
	// turn leaves the checkpoint alone, so the next pass still covers it.
	if (!hasTools || !turnText.trim()) {
		return;
	}

	const runner = resolveRunner();
	if (!runner) {
		logNote(`session=${sessionId} skipped: direct runner not found (tried ${RUNNER_CANDIDATES.join(", ")})`);
		return;
	}
	const payloadFile = join(HOOK_DIR, `pi-${sessionId}-${Date.now()}.json`);

	try {
		mkdirSync(HOOK_DIR, { recursive: true });
		writeFileSync(
			payloadFile,
			JSON.stringify({
				session_id: sessionId,
				cwd: ctx.cwd,
				turn_text: turnText,
				pi_node: process.execPath,
				pi_cli: process.argv[1],
				pi_extension: thisExtensionPath(),
				pi_model: ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined,
			})
		);

		const child = spawn(process.execPath, [runner, payloadFile], {
			detached: true,
			stdio: "ignore",
			env: {
				...process.env,
				BIGBRAIN_MAINT: "1",
			},
		});

		child.unref();
		// Advance only once the pass is handed off, so a failed spawn is retried next time.
		if (lastEntryId) writeCheckpoint(sessionId, lastEntryId);
	} catch (err: any) {
		// A failed trigger must never crash the interactive session.
		logNote(`session=${sessionId} FAILED to spawn direct runner: ${err?.message || String(err)}`);
	}
}

export default function bigbrainExtension(pi: ExtensionAPI) {
	// Register background maintenance hook on turn completion
	pi.on("agent_settled", async (_event, ctx) => {
		triggerDetachedMaintenance(ctx);
	});

	// memory_recall
	pi.registerTool({
		name: "memory_recall",
		label: "Memory Recall",
		description: "Recall relevant knowledge by meaning. Provide a natural-language query describing what you want to remember; results are ranked by a blend of semantic similarity, recency, and importance. Optionally filter by tags or source. Call this proactively at the start of a task to check what is already known.",
		parameters: Type.Object({
			query: Type.String({ description: "Natural-language query describing what you want to recall" }),
			limit: Type.Optional(Type.Integer({ description: "Max results to return (default 5)", default: 5 })),
			tags: Type.Optional(Type.Array(Type.String(), { description: "Filter results by tags" })),
			source: Type.Optional(Type.String({ description: "Filter results by source" })),
			min_similarity: Type.Optional(Type.Number({ description: "Minimum similarity threshold (0.0 - 1.0)", default: 0.0 })),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_recall", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_store
	pi.registerTool({
		name: "memory_store",
		label: "Memory Store",
		description: "Store a durable piece of knowledge in long-term memory. `topic` is a short semantic key (it becomes the searchable embedding); `content` is the detailed knowledge. A near-duplicate topic updates the existing memory instead of creating a duplicate. Use this to remember decisions, facts, preferences, and learnings worth recalling later. If the result's action is 'needs_rewrite', nothing was written: the existing entry is too large to keep appending to, so re-store the same topic with on_conflict='replace' and a compact current-truth rewrite that folds in the new learning.",
		parameters: Type.Object({
			topic: Type.String({ description: "Short, descriptive, searchable key (becomes the embedding)" }),
			content: Type.String({ description: "Full detail, self-contained knowledge" }),
			tags: Type.Optional(Type.Array(Type.String(), { description: "Lowercase, reusable tags" })),
			source: Type.Optional(Type.String({ description: "Source of the knowledge", default: "" })),
			importance: Type.Optional(Type.Number({ description: "Importance score 0.0 - 1.0 (default 0.5)", default: 0.5 })),
			on_conflict: Type.Optional(Type.String({ description: "'auto' (append while small, else ask for a rewrite), 'replace' (overwrite with this content), 'merge' (always append), 'skip', or 'new'", default: "auto" })),
			dedup: Type.Optional(Type.Boolean({ description: "Whether to deduplicate against similar topics", default: true })),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_store", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_get
	pi.registerTool({
		name: "memory_get",
		label: "Memory Get",
		description: "Fetch a single memory by its id. Returns null if not found.",
		parameters: Type.Object({
			memory_id: Type.String({ description: "ID of the memory to fetch, as a string (ids exceed JS safe-integer range)" }),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_get", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_list
	pi.registerTool({
		name: "memory_list",
		label: "Memory List",
		description: "List stored memories, most recently updated first. Optionally filter by tags or source. Useful for browsing rather than semantic search.",
		parameters: Type.Object({
			limit: Type.Optional(Type.Integer({ description: "Max memories to return (default 50)", default: 50 })),
			offset: Type.Optional(Type.Integer({ description: "Offset for pagination (default 0)", default: 0 })),
			tags: Type.Optional(Type.Array(Type.String(), { description: "Filter by tags" })),
			source: Type.Optional(Type.String({ description: "Filter by source" })),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_list", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_update
	pi.registerTool({
		name: "memory_update",
		label: "Memory Update",
		description: "Update fields of an existing memory by id. Only provided fields change; changing the topic re-embeds the search key.",
		parameters: Type.Object({
			memory_id: Type.String({ description: "ID of the memory to update, as a string (ids exceed JS safe-integer range)" }),
			topic: Type.Optional(Type.String({ description: "New topic" })),
			content: Type.Optional(Type.String({ description: "New content" })),
			tags: Type.Optional(Type.Array(Type.String(), { description: "New tags" })),
			source: Type.Optional(Type.String({ description: "New source" })),
			importance: Type.Optional(Type.Number({ description: "New importance score 0.0 - 1.0" })),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_update", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_delete
	pi.registerTool({
		name: "memory_delete",
		label: "Memory Delete",
		description: "Delete one or more memories by id. Returns the number that actually existed and were deleted.",
		parameters: Type.Object({
			memory_ids: Type.Array(Type.String(), { description: "List of memory IDs to delete, as strings (ids exceed JS safe-integer range)" }),
		}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_delete", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});

	// memory_count
	pi.registerTool({
		name: "memory_count",
		label: "Memory Count",
		description: "Return the total number of stored memories.",
		parameters: Type.Object({}),
		async execute(_toolCallId, params, signal) {
			const res = await callMcpTool("memory_count", params, signal);
			return {
				content: res.content,
				details: {},
			};
		},
	});
}
