use std::net::TcpListener;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use serde::Serialize;

pub struct BackendState {
    pub child: Mutex<Option<Child>>,
    pub port: Mutex<u16>,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            child: Mutex::new(None),
            port: Mutex::new(0),
        }
    }
}

#[derive(Clone, Serialize)]
struct BackendReadyPayload {
    port: u16,
}

/// Find an available port by binding to port 0
fn find_available_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to find available port: {}", e))?;
    let port = listener.local_addr()
        .map_err(|e| format!("Failed to get local address: {}", e))?
        .port();
    Ok(port)
}

/// Start the Python backend sidecar
pub fn start_backend(app_handle: &AppHandle) -> Result<(), String> {
    let state = app_handle.state::<BackendState>();

    let port = find_available_port()?;
    *state.port.lock().unwrap() = port;

    let is_dev = std::env::var("STOCKPILE_DEV").unwrap_or_default() == "1";

    let child = if is_dev {
        // Dev mode: run Python directly from project root
        // Tauri dev runs from web/src-tauri, so go up two levels
        let project_root = std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?;

        let root = project_root
            .ancestors()
            .find(|p| p.join("src").join("api").join("server.py").exists())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| project_root.clone());

        // Use venv python if available, otherwise system python3/python
        let venv_python = root.join(".venv").join("bin").join("python");
        let python = if venv_python.exists() {
            venv_python.to_str().unwrap_or("python3").to_string()
        } else if root.join("venv").join("bin").join("python").exists() {
            root.join("venv").join("bin").join("python").to_str().unwrap_or("python3").to_string()
        } else {
            "python3".to_string()
        };

        Command::new(&python)
            .args([
                "-m", "uvicorn",
                "src.api.server:create_app",
                "--factory",
                "--host", "127.0.0.1",
                "--port", &port.to_string(),
            ])
            .current_dir(&root)
            .env("PORT", port.to_string())
            .spawn()
            .map_err(|e| format!("Failed to start dev backend (tried {} from {:?}): {}", python, root, e))?
    } else {
        // Production mode: run sidecar binary
        let sidecar_path = app_handle.path()
            .resource_dir()
            .map_err(|e| format!("Failed to get resource dir: {}", e))?
            .join("binaries")
            .join(format!("stockpile-backend-{}-apple-darwin", std::env::consts::ARCH));

        Command::new(&sidecar_path)
            .env("PORT", port.to_string())
            .spawn()
            .map_err(|e| format!("Failed to start sidecar at {:?}: {}", sidecar_path, e))?
    };

    *state.child.lock().unwrap() = Some(child);

    // Wait for health check in a background thread
    let handle = app_handle.clone();
    let health_port = port;
    std::thread::spawn(move || {
        match wait_for_health(health_port, Duration::from_secs(30)) {
            Ok(()) => {
                let _ = handle.emit("backend-ready", BackendReadyPayload { port: health_port });
                println!("Backend ready on port {}", health_port);
            }
            Err(e) => {
                eprintln!("Backend health check failed: {}", e);
            }
        }
    });

    Ok(())
}

/// Poll the health endpoint until it responds
fn wait_for_health(port: u16, timeout: Duration) -> Result<(), String> {
    let start = std::time::Instant::now();
    let url = format!("http://127.0.0.1:{}/api/health", port);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    loop {
        if start.elapsed() > timeout {
            return Err(format!("Backend did not become healthy within {:?}", timeout));
        }

        match client.get(&url).send() {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {}
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Stop the backend process
pub fn stop_backend(app_handle: &AppHandle) {
    let state = app_handle.state::<BackendState>();
    let mut guard = state.child.lock().unwrap();
    if let Some(mut child) = guard.take() {
        drop(guard);
        // Try graceful shutdown first
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(child.id() as i32, libc::SIGTERM);
            }
        }

        // Wait briefly for graceful shutdown
        std::thread::sleep(Duration::from_secs(2));

        // Force kill if still running
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
pub fn get_backend_port(state: tauri::State<BackendState>) -> u16 {
    *state.port.lock().unwrap()
}

#[tauri::command]
pub fn check_backend_health(state: tauri::State<BackendState>) -> bool {
    let port = *state.port.lock().unwrap();
    if port == 0 {
        return false;
    }

    let url = format!("http://127.0.0.1:{}/api/health", port);
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };

    matches!(client.get(&url).send(), Ok(resp) if resp.status().is_success())
}
