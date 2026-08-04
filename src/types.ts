export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

/** Per-model sampling settings. Field names match Ollama's `options` keys,
 * which is what the backend forwards them as. Every key is optional: an absent
 * one means "leave the engine on its own default" rather than zero. */
export interface GenerationParams {
  num_ctx?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  repeat_last_n?: number;
  repeat_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
}

export interface ModelInfo {
  name: string;
  size_bytes: number | null;
  parameter_size: string | null;
  quantization: string | null;
  supports_thinking: boolean;
}

export type ChatStreamEvent =
  | { type: "queued"; request_id: string }
  | { type: "thinking"; request_id: string; content: string }
  | { type: "token"; request_id: string; content: string }
  | { type: "done"; request_id: string }
  | { type: "cancelled"; request_id: string }
  | { type: "error"; request_id: string; message: string };

export type PackageManager = "winget" | "homebrew";

export interface EngineVersion {
  version: string;
  /** Whether the engine is new enough for the API surface this app uses. */
  supported: boolean;
  minimum: string;
  /** Newest published release, or null when it couldn't be looked up. */
  latest: string | null;
  update_available: boolean;
  /** How an in-app update would run, or null if it can't on this machine. */
  package_manager: PackageManager | null;
  download_page: string;
}

export type UpdateProgressEvent =
  | { type: "line"; text: string }
  | { type: "done"; success: boolean; message: string };

export interface ProviderInfo {
  id: string;
  label: string;
  default_model: string;
  configured: boolean;
}

export interface ModelTarget {
  id: string;
  provider: string;
  model: string;
  label: string;
}

export interface HardwareProfile {
  os: string;
  total_ram_gb: number;
  logical_cores: number;
  accelerated: boolean;
  /** Weight-read throughput learned from generation the engine has actually
   * done on this machine. null until the first local answer completes. */
  observed_gb_per_sec: number | null;
}

export interface CatalogEntry {
  tag: string;
  origin: string;
  label: string;
  size_gb: number;
  active_size_gb: number;
  role: string;
  description: string;
  /** null when a GPU is doing the work, where a bandwidth-derived estimate
   * would be meaningless. */
  est_tokens_per_sec: number | null;
}

export type PullProgressEvent =
  | { type: "progress"; model: string; status: string; completed: number | null; total: number | null }
  | { type: "done"; model: string }
  | { type: "error"; model: string; message: string };

export interface RouterDecision {
  providers: string[];
  compressed_prompt: string;
  /** Decided without a model call, because there was nothing to compress. */
  shortcut: boolean;
}
