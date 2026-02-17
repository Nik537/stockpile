// Content block types from Claude CLI stream-json output

export interface TextBlock {
  type: 'text'
  text: string
}

export interface ToolUseBlock {
  type: 'tool_use'
  id: string
  name: string
  input: Record<string, any>
}

export interface ToolResultBlock {
  type: 'tool_result'
  tool_use_id: string
  content: string
}

export type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

// Claude CLI stream-json message types

export interface ClaudeAssistantMessage {
  type: 'assistant'
  message: {
    role: string
    content: ContentBlock[]
    model: string
    stop_reason?: string
  }
}

export interface ClaudeResultMessage {
  type: 'result'
  result: {
    role: string
    content: ContentBlock[]
    model: string
    stop_reason: string
    usage: { input_tokens: number; output_tokens: number }
  }
  session_id: string
}

export interface ClaudeErrorMessage {
  type: 'error'
  error: { message: string; type: string }
}

export type ClaudeStreamMessage =
  | ClaudeAssistantMessage
  | ClaudeResultMessage
  | ClaudeErrorMessage

// UI-level types for conversation display

export interface ToolCall {
  id: string
  name: string
  input: Record<string, any>
  output?: string
  status: 'running' | 'completed' | 'error'
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  toolCalls?: ToolCall[]
  isStreaming?: boolean
}
