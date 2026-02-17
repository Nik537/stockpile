import { useEffect, useCallback, useRef } from 'react'
import { useScriptwriterStore } from '../stores/useScriptwriterStore'
import { isTauri } from '../lib/tauri'
import type { ClaudeStreamMessage, ContentBlock } from '../lib/claude-messages'

const SESSION_STORAGE_KEY = 'scriptwriter-session-id'

interface UseClaudeSessionConfig {
  workingDir: string
  systemPrompt: string
}

function extractText(content: ContentBlock[]): string {
  return content
    .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
    .map((b) => b.text)
    .join('')
}

// Grab stable action references once — these never change between renders
const actions = () => useScriptwriterStore.getState()

export function useClaudeSession({ workingDir, systemPrompt }: UseClaudeSessionConfig) {
  const sessionIdRef = useRef<string | null>(null)
  const unlistenersRef = useRef<(() => void)[]>([])
  const isConnectedRef = useRef(false)
  const spawnedRef = useRef(false) // guard against double-spawn

  const spawnSession = useCallback(
    async (sessionId: string) => {
      if (!isTauri() || spawnedRef.current) return
      spawnedRef.current = true

      try {
        const { invoke } = await import('@tauri-apps/api/core')
        const { listen } = await import('@tauri-apps/api/event')

        await invoke('spawn_claude_json', {
          sessionId,
          workingDir,
          systemPrompt,
          model: useScriptwriterStore.getState().model,
        })

        isConnectedRef.current = true
        actions().setSessionId(sessionId)
        actions().setError(null)

        const unlistenMessage = await listen<{ session_id: string; data: string }>(
          'claude-message',
          (event) => {
            if (event.payload.session_id !== sessionIdRef.current) return

            let parsed: ClaudeStreamMessage
            try {
              parsed = JSON.parse(event.payload.data)
            } catch {
              return
            }

            const s = actions()

            if (parsed.type === 'assistant') {
              const text = extractText(parsed.message.content)
              const toolCalls = parsed.message.content
                .filter(
                  (b): b is { type: 'tool_use'; id: string; name: string; input: Record<string, any> } =>
                    b.type === 'tool_use'
                )
                .map((b) => ({ id: b.id, name: b.name, input: b.input, status: 'running' as const }))

              const toolResults = parsed.message.content.filter(
                (b): b is { type: 'tool_result'; tool_use_id: string; content: string } =>
                  b.type === 'tool_result'
              )

              for (const tc of toolCalls) {
                s.addToolCall(tc)
              }
              for (const tr of toolResults) {
                s.updateToolCall(tr.tool_use_id, { output: tr.content, status: 'completed' })
              }

              const msgs = useScriptwriterStore.getState().messages
              const lastMsg = msgs[msgs.length - 1]
              if (lastMsg && lastMsg.role === 'assistant' && lastMsg.isStreaming) {
                s.updateLastAssistantMessage(text, toolCalls.length > 0 ? toolCalls : undefined)
              } else if (text || toolCalls.length > 0) {
                s.addMessage({
                  id: crypto.randomUUID(),
                  role: 'assistant',
                  content: text,
                  timestamp: Date.now(),
                  toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
                  isStreaming: true,
                })
              }
            } else if (parsed.type === 'result') {
              const text = extractText(parsed.result.content)
              const msgs = useScriptwriterStore.getState().messages
              const lastMsg = msgs[msgs.length - 1]

              if (lastMsg && lastMsg.role === 'assistant' && lastMsg.isStreaming) {
                s.updateLastAssistantMessage(text)
                useScriptwriterStore.setState((prev) => {
                  const messages = [...prev.messages]
                  for (let i = messages.length - 1; i >= 0; i--) {
                    if (messages[i].role === 'assistant' && messages[i].isStreaming) {
                      messages[i] = { ...messages[i], isStreaming: false }
                      break
                    }
                  }
                  return { messages }
                })
              } else if (text) {
                s.addMessage({
                  id: crypto.randomUUID(),
                  role: 'assistant',
                  content: text,
                  timestamp: Date.now(),
                  isStreaming: false,
                })
              }

              // Mark all running tools as completed
              const activity = useScriptwriterStore.getState().toolActivity
              for (const tc of activity) {
                if (tc.status === 'running') {
                  s.updateToolCall(tc.id, { status: 'completed' })
                }
              }

              s.setGenerating(false)

              if (parsed.session_id) {
                s.setSessionId(parsed.session_id)
                localStorage.setItem(SESSION_STORAGE_KEY, parsed.session_id)
              }
            } else if (parsed.type === 'error') {
              s.setError(parsed.error.message)
              s.setGenerating(false)
            }
          }
        )

        const unlistenExit = await listen<{ session_id: string; code: number | null }>(
          'claude-exit',
          (event) => {
            if (event.payload.session_id !== sessionIdRef.current) return
            isConnectedRef.current = false
            spawnedRef.current = false
            actions().setGenerating(false)
          }
        )

        unlistenersRef.current = [unlistenMessage, unlistenExit]
      } catch (e: any) {
        const msg = typeof e === 'string' ? e : e?.message || 'Failed to start Claude session'
        actions().setError(msg)
        isConnectedRef.current = false
        spawnedRef.current = false
      }
    },
    // Only re-create if config actually changes — NOT on store changes
    [workingDir, systemPrompt]
  )

  const killSession = useCallback(async () => {
    for (const unlisten of unlistenersRef.current) {
      unlisten()
    }
    unlistenersRef.current = []

    const sid = sessionIdRef.current
    if (sid && isTauri()) {
      try {
        const { invoke } = await import('@tauri-apps/api/core')
        await invoke('kill_claude_json', { sessionId: sid })
      } catch {
        // Ignore kill errors
      }
    }

    isConnectedRef.current = false
    spawnedRef.current = false
    sessionIdRef.current = null
  }, [])

  // Mount: spawn session (runs once)
  useEffect(() => {
    if (!isTauri()) return

    const savedId = localStorage.getItem(SESSION_STORAGE_KEY)
    const sessionId = savedId || crypto.randomUUID()
    sessionIdRef.current = sessionId
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId)

    spawnSession(sessionId)

    return () => {
      killSession()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const sendMessage = useCallback(async (text: string) => {
    if (!sessionIdRef.current || !isTauri()) return

    actions().addMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    })
    actions().setGenerating(true)
    actions().setError(null)

    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('send_claude_message', {
        sessionId: sessionIdRef.current,
        message: text,
      })
    } catch (e: any) {
      const msg = typeof e === 'string' ? e : e?.message || 'Failed to send message'
      actions().setError(msg)
      actions().setGenerating(false)
    }
  }, [])

  const stopGeneration = useCallback(async () => {
    if (!sessionIdRef.current || !isTauri()) return

    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('abort_claude_generation', { sessionId: sessionIdRef.current })
    } catch {
      // Ignore abort errors
    }

    actions().setGenerating(false)
  }, [])

  const refineSelection = useCallback(async (messageId: string, selectedText: string) => {
    if (!sessionIdRef.current || !isTauri()) return

    const prompt = `Rewrite the following selected text, keeping the same tone and style. Only output the replacement text, nothing else:\n\n${selectedText}`

    actions().setGenerating(true)
    actions().setError(null)

    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('send_claude_message', {
        sessionId: sessionIdRef.current,
        message: prompt,
      })
    } catch (e: any) {
      const msg = typeof e === 'string' ? e : e?.message || 'Failed to refine selection'
      actions().setError(msg)
      actions().setGenerating(false)
    }
  }, [])

  const newSession = useCallback(async () => {
    await killSession()

    actions().clearMessages()
    actions().setSessionId(null)

    const sessionId = crypto.randomUUID()
    sessionIdRef.current = sessionId
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId)

    await spawnSession(sessionId)
  }, [killSession, spawnSession])

  return {
    sendMessage,
    stopGeneration,
    newSession,
    refineSelection,
    isConnected: isConnectedRef.current,
  }
}
