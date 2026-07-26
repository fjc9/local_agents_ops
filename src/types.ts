export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ModelInfo {
  name: string;
  size_bytes: number | null;
  parameter_size: string | null;
  quantization: string | null;
}

export type ChatStreamEvent =
  | { type: "thinking"; request_id: string; content: string }
  | { type: "token"; request_id: string; content: string }
  | { type: "done"; request_id: string }
  | { type: "error"; request_id: string; message: string };

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
}

export interface CatalogEntry {
  tag: string;
  origin: string;
  label: string;
  size_gb: number;
  role: string;
  description: string;
}

export type PullProgressEvent =
  | { type: "progress"; model: string; status: string; completed: number | null; total: number | null }
  | { type: "done"; model: string }
  | { type: "error"; model: string; message: string };
