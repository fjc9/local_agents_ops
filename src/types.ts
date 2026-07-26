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
