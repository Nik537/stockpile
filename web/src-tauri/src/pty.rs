use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use portable_pty::{CommandBuilder, MasterPty, NativePtySystem, PtySize, PtySystem};
use tauri::{AppHandle, Emitter};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Serialize;

pub(crate) struct PtySession {
    master: Arc<Mutex<Box<dyn MasterPty + Send>>>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    alive: Arc<Mutex<bool>>,
}

pub struct PtyState {
    pub(crate) sessions: Mutex<HashMap<String, PtySession>>,
}

impl Default for PtyState {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

#[derive(Clone, Serialize)]
struct PtyDataPayload {
    session_id: String,
    data: String,
}

#[derive(Clone, Serialize)]
struct PtyExitPayload {
    session_id: String,
    code: Option<u32>,
}

#[tauri::command]
pub fn spawn_claude(
    session_id: String,
    working_dir: String,
    app_handle: AppHandle,
    state: tauri::State<PtyState>,
) -> Result<(), String> {
    // Check if session already exists
    {
        let sessions = state.sessions.lock().unwrap();
        if sessions.contains_key(&session_id) {
            return Err(format!("Session {} already exists", session_id));
        }
    }

    let pty_system = NativePtySystem::default();

    let pair = pty_system
        .openpty(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("Failed to open PTY: {}", e))?;

    println!("[PTY] spawn_claude called: session={}, dir={}", session_id, working_dir);

    // Resolve claude binary - portable-pty doesn't inherit shell PATH
    let claude_bin = std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .map(std::path::PathBuf::from)
        .find(|p| p.join("claude").exists())
        .map(|p| p.join("claude").to_string_lossy().to_string())
        .unwrap_or_else(|| "claude".to_string());

    println!("[PTY] Resolved claude binary: {}", claude_bin);

    let mut cmd = CommandBuilder::new(&claude_bin);
    cmd.cwd(&working_dir);

    // Inherit environment so claude has access to API keys, PATH, etc.
    for (key, value) in std::env::vars() {
        cmd.env(key, value);
    }

    let child = pair.slave
        .spawn_command(cmd)
        .map_err(|e| {
            let msg = e.to_string();
            eprintln!("[PTY] Failed to spawn: {}", msg);
            if msg.contains("No such file") || msg.contains("not found") {
                "Claude Code CLI not found. Install it with: npm install -g @anthropic-ai/claude-code".to_string()
            } else {
                format!("Failed to spawn claude: {}", msg)
            }
        })?;

    println!("[PTY] Claude process spawned successfully");

    drop(pair.slave);

    let writer = pair.master
        .take_writer()
        .map_err(|e| format!("Failed to get PTY writer: {}", e))?;

    let mut reader = pair.master
        .try_clone_reader()
        .map_err(|e| format!("Failed to get PTY reader: {}", e))?;

    let master = Arc::new(Mutex::new(pair.master));
    let writer = Arc::new(Mutex::new(writer));
    let alive = Arc::new(Mutex::new(true));

    // Reader thread - emits pty-data events
    let sid = session_id.clone();
    let handle = app_handle.clone();
    let alive_clone = alive.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let encoded = BASE64.encode(&buf[..n]);
                    let _ = handle.emit("pty-data", PtyDataPayload {
                        session_id: sid.clone(),
                        data: encoded,
                    });
                }
                Err(_) => break,
            }
        }
        *alive_clone.lock().unwrap() = false;
    });

    // Exit watcher thread
    let exit_sid = session_id.clone();
    let exit_handle = app_handle.clone();
    let mut child = child;
    std::thread::spawn(move || {
        let status = child.wait();
        let code = status.ok().map(|s| s.exit_code());
        let _ = exit_handle.emit("pty-exit", PtyExitPayload {
            session_id: exit_sid,
            code,
        });
    });

    // Store session
    let session = PtySession {
        master,
        writer,
        alive,
    };

    state.sessions.lock().unwrap().insert(session_id, session);

    Ok(())
}

#[tauri::command]
pub fn write_to_pty(
    session_id: String,
    data: String,
    state: tauri::State<PtyState>,
) -> Result<(), String> {
    let sessions = state.sessions.lock().unwrap();
    let session = sessions.get(&session_id)
        .ok_or_else(|| format!("No PTY session: {}", session_id))?;

    let mut writer = session.writer.lock().unwrap();
    writer.write_all(data.as_bytes())
        .map_err(|e| format!("Write failed: {}", e))?;
    writer.flush()
        .map_err(|e| format!("Flush failed: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn resize_pty(
    session_id: String,
    cols: u16,
    rows: u16,
    state: tauri::State<PtyState>,
) -> Result<(), String> {
    let sessions = state.sessions.lock().unwrap();
    let session = sessions.get(&session_id)
        .ok_or_else(|| format!("No PTY session: {}", session_id))?;

    let master = session.master.lock().unwrap();
    master.resize(PtySize {
        rows,
        cols,
        pixel_width: 0,
        pixel_height: 0,
    }).map_err(|e| format!("Resize failed: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn kill_pty(
    session_id: String,
    state: tauri::State<PtyState>,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().unwrap();
    if let Some(session) = sessions.remove(&session_id) {
        *session.alive.lock().unwrap() = false;
        // Dropping the master/writer will close the PTY
        drop(session);
    }
    Ok(())
}
