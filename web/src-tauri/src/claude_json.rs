use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

pub struct ClaudeJsonSession {
    stdin: Arc<Mutex<ChildStdin>>,
    alive: Arc<Mutex<bool>>,
    pid: u32,
}

pub struct ClaudeJsonState {
    pub(crate) sessions: Mutex<HashMap<String, ClaudeJsonSession>>,
}

impl Default for ClaudeJsonState {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

#[derive(Clone, Serialize)]
struct ClaudeMessagePayload {
    session_id: String,
    data: String,
}

#[derive(Clone, Serialize)]
struct ClaudeExitPayload {
    session_id: String,
    code: Option<i32>,
}

/// Resolve the claude binary by scanning PATH entries.
fn resolve_claude_binary() -> String {
    std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .map(std::path::PathBuf::from)
        .find(|p| p.join("claude").exists())
        .map(|p| p.join("claude").to_string_lossy().to_string())
        .unwrap_or_else(|| "claude".to_string())
}

#[tauri::command]
pub fn spawn_claude_json(
    session_id: String,
    working_dir: String,
    system_prompt: Option<String>,
    model: Option<String>,
    app_handle: AppHandle,
    state: tauri::State<ClaudeJsonState>,
) -> Result<(), String> {
    // Check if session already exists
    {
        let sessions = state.sessions.lock().unwrap();
        if sessions.contains_key(&session_id) {
            return Err(format!("Session {} already exists", session_id));
        }
    }

    let claude_bin = resolve_claude_binary();
    println!(
        "[ClaudeJSON] spawn_claude_json called: session={}, dir={}, binary={}",
        session_id, working_dir, claude_bin
    );

    let mut cmd = Command::new(&claude_bin);
    cmd.current_dir(&working_dir)
        .arg("--print")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--include-partial-messages")
        .arg("--permission-mode")
        .arg("acceptEdits")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref prompt) = system_prompt {
        cmd.arg("--append-system-prompt").arg(prompt);
    }

    if let Some(ref m) = model {
        cmd.arg("--model").arg(m);
    }

    // Clear inherited env and re-add everything EXCEPT CLAUDECODE
    // to avoid "nested session" detection when launched from within
    // a Claude Code session (e.g. during development).
    cmd.env_clear();
    cmd.envs(
        std::env::vars().filter(|(key, _)| key != "CLAUDECODE")
    );

    let mut child: Child = cmd.spawn().map_err(|e| {
        let msg = e.to_string();
        eprintln!("[ClaudeJSON] Failed to spawn: {}", msg);
        if msg.contains("No such file") || msg.contains("not found") {
            "Claude Code CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
                .to_string()
        } else {
            format!("Failed to spawn claude: {}", msg)
        }
    })?;

    let pid = child.id();
    println!("[ClaudeJSON] Claude process spawned with PID {}", pid);

    let stdin = child
        .stdin
        .take()
        .ok_or("Failed to capture stdin")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Failed to capture stdout")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Failed to capture stderr")?;

    let stdin = Arc::new(Mutex::new(stdin));
    let alive = Arc::new(Mutex::new(true));

    // Stdout reader thread — emits claude-message events (one per JSON line)
    let sid = session_id.clone();
    let handle = app_handle.clone();
    let alive_stdout = alive.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    let _ = handle.emit(
                        "claude-message",
                        ClaudeMessagePayload {
                            session_id: sid.clone(),
                            data: text,
                        },
                    );
                }
                Err(_) => break,
            }
        }
        *alive_stdout.lock().unwrap() = false;
    });

    // Stderr reader thread — log stderr output for debugging
    let stderr_sid = session_id.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    if !text.is_empty() {
                        eprintln!("[ClaudeJSON][{}] stderr: {}", stderr_sid, text);
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Exit watcher thread
    let exit_sid = session_id.clone();
    let exit_handle = app_handle.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        let code = status.ok().and_then(|s| s.code());
        println!("[ClaudeJSON] Process exited: session={}, code={:?}", exit_sid, code);
        let _ = exit_handle.emit(
            "claude-exit",
            ClaudeExitPayload {
                session_id: exit_sid,
                code,
            },
        );
    });

    // Store session
    let session = ClaudeJsonSession {
        stdin,
        alive,
        pid,
    };

    state.sessions.lock().unwrap().insert(session_id, session);

    Ok(())
}

#[tauri::command]
pub fn send_claude_message(
    session_id: String,
    message: String,
    state: tauri::State<ClaudeJsonState>,
) -> Result<(), String> {
    let sessions = state.sessions.lock().unwrap();
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("No Claude JSON session: {}", session_id))?;

    // Check if still alive
    if !*session.alive.lock().unwrap() {
        return Err(format!("Session {} has exited", session_id));
    }

    // Build the JSON input message in Claude CLI stream-json format
    let input = serde_json::json!({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": message}]
        }
    });
    let mut payload = serde_json::to_string(&input)
        .map_err(|e| format!("JSON serialize failed: {}", e))?;
    payload.push('\n');

    let mut stdin = session.stdin.lock().unwrap();
    stdin
        .write_all(payload.as_bytes())
        .map_err(|e| format!("Write to stdin failed: {}", e))?;
    stdin
        .flush()
        .map_err(|e| format!("Flush stdin failed: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn abort_claude_generation(
    session_id: String,
    state: tauri::State<ClaudeJsonState>,
) -> Result<(), String> {
    let sessions = state.sessions.lock().unwrap();
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("No Claude JSON session: {}", session_id))?;

    // Send SIGINT to interrupt current generation without killing the process
    let pid = session.pid as libc::pid_t;
    println!("[ClaudeJSON] Sending SIGINT to PID {} (session={})", pid, session_id);
    let ret = unsafe { libc::kill(pid, libc::SIGINT) };
    if ret != 0 {
        return Err(format!(
            "Failed to send SIGINT to PID {}: errno {}",
            pid,
            std::io::Error::last_os_error()
        ));
    }

    Ok(())
}

#[tauri::command]
pub fn kill_claude_json(
    session_id: String,
    state: tauri::State<ClaudeJsonState>,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().unwrap();
    if let Some(session) = sessions.remove(&session_id) {
        *session.alive.lock().unwrap() = false;

        // Send SIGKILL to terminate the process
        let pid = session.pid as libc::pid_t;
        println!("[ClaudeJSON] Killing PID {} (session={})", pid, session_id);
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }

        // Drop session resources (closes stdin which also signals EOF to process)
        drop(session);
    }
    Ok(())
}
