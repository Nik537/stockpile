#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};

mod sidecar;
mod pty;
mod claude_json;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(sidecar::BackendState::default())
        .manage(pty::PtyState::default())
        .manage(claude_json::ClaudeJsonState::default())
        .setup(|app| {
            let handle = app.handle().clone();

            // Open devtools in debug mode
            #[cfg(debug_assertions)]
            if let Some(window) = app.get_webview_window("main") {
                window.open_devtools();
            }

            // Start backend sidecar
            if let Err(e) = sidecar::start_backend(&handle) {
                eprintln!("Failed to start backend: {}", e);
            }

            // Build native macOS menu bar
            let menu = Menu::with_items(&handle, &[
                &Submenu::with_items(&handle, "Stockpile", true, &[
                    &PredefinedMenuItem::about(&handle, Some("About Stockpile"), None)?,
                    &PredefinedMenuItem::separator(&handle)?,
                    &PredefinedMenuItem::quit(&handle, Some("Quit Stockpile"))?,
                ])?,
                &Submenu::with_items(&handle, "Edit", true, &[
                    &PredefinedMenuItem::undo(&handle, None)?,
                    &PredefinedMenuItem::redo(&handle, None)?,
                    &PredefinedMenuItem::separator(&handle)?,
                    &PredefinedMenuItem::cut(&handle, None)?,
                    &PredefinedMenuItem::copy(&handle, None)?,
                    &PredefinedMenuItem::paste(&handle, None)?,
                    &PredefinedMenuItem::select_all(&handle, None)?,
                ])?,
                &Submenu::with_items(&handle, "View", true, &[
                    &MenuItem::with_id(&handle, "reload", "Reload", true, Some("CmdOrCtrl+R"))?,
                    &PredefinedMenuItem::separator(&handle)?,
                    &PredefinedMenuItem::fullscreen(&handle, None)?,
                ])?,
                &Submenu::with_items(&handle, "Window", true, &[
                    &PredefinedMenuItem::minimize(&handle, None)?,
                    &PredefinedMenuItem::separator(&handle)?,
                    &PredefinedMenuItem::close_window(&handle, Some("Close"))?,
                ])?,
            ])?;

            app.set_menu(menu)?;

            Ok(())
        })
        .on_menu_event(|app, event| {
            if event.id() == "reload" {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.eval("location.reload()");
                }
            }
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                sidecar::stop_backend(&window.app_handle());
            }
        })
        .invoke_handler(tauri::generate_handler![
            sidecar::get_backend_port,
            sidecar::check_backend_health,
            pty::spawn_claude,
            pty::write_to_pty,
            pty::resize_pty,
            pty::kill_pty,
            claude_json::spawn_claude_json,
            claude_json::send_claude_message,
            claude_json::abort_claude_generation,
            claude_json::kill_claude_json,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Stockpile");
}
