import { create } from 'zustand'
import type { Message, ToolCall } from '../lib/claude-messages'

interface ScriptwriterState {
  messages: Message[]
  isGenerating: boolean
  sessionId: string | null
  model: string
  error: string | null
  toolActivity: ToolCall[]
  addMessage: (msg: Message) => void
  updateLastAssistantMessage: (content: string, toolCalls?: ToolCall[]) => void
  setGenerating: (generating: boolean) => void
  clearMessages: () => void
  addToolCall: (tc: ToolCall) => void
  updateToolCall: (id: string, updates: Partial<ToolCall>) => void
  updateMessageContent: (id: string, content: string) => void
  replaceTextInMessage: (id: string, oldText: string, newText: string) => void
  deleteMessage: (id: string) => void
  setError: (error: string | null) => void
  setSessionId: (id: string | null) => void
  setModel: (model: string) => void
}

export const useScriptwriterStore = create<ScriptwriterState>((set) => ({
  messages: [],
  isGenerating: false,
  sessionId: null,
  model: 'sonnet',
  error: null,
  toolActivity: [],
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  updateLastAssistantMessage: (content, toolCalls) =>
    set((state) => {
      const messages = [...state.messages]
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'assistant') {
          messages[i] = {
            ...messages[i],
            content,
            ...(toolCalls !== undefined ? { toolCalls } : {}),
          }
          break
        }
      }
      return { messages }
    }),
  setGenerating: (generating) => set({ isGenerating: generating }),
  clearMessages: () => set({ messages: [], toolActivity: [], error: null }),
  addToolCall: (tc) => set((state) => ({ toolActivity: [...state.toolActivity, tc] })),
  updateToolCall: (id, updates) =>
    set((state) => ({
      toolActivity: state.toolActivity.map((tc) =>
        tc.id === id ? { ...tc, ...updates } : tc
      ),
    })),
  updateMessageContent: (id, content) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content } : msg
      ),
    })),
  replaceTextInMessage: (id, oldText, newText) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content: msg.content.replace(oldText, newText) } : msg
      ),
    })),
  deleteMessage: (id) =>
    set((state) => ({
      messages: state.messages.filter((msg) => msg.id !== id),
    })),
  setError: (error) => set({ error }),
  setSessionId: (id) => set({ sessionId: id }),
  setModel: (model) => set({ model }),
}))
