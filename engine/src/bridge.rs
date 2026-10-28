//! Persistent local subprocess bridge: Rust owns the world, Python owns the brain.
//! JSON-lines protocol v1, ordered command/response, one outstanding request.
//! No external network ports or services are opened.

use crate::world::MotorCommand;
use serde::{Deserialize, Serialize};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread,
};

const VERSION: u8 = 1;

#[derive(Debug, Serialize)]
struct Step {
    #[serde(rename = "type")]
    kind: &'static str,
    version: u8,
    seq: u64,
    left: f32,
    right: f32,
    facets: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Reply {
    #[serde(rename = "ready")]
    Ready { version: u8, mode: String, neurons: usize, directed_pairs: usize },
    #[serde(rename = "motor")]
    Motor { version: u8, seq: u64, left: u8, right: u8, active: usize },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "saved")]
    Saved {},
}

#[derive(Debug)]
pub enum Event {
    Ready { mode: String, neurons: usize, edges: usize },
    Motor { command: MotorCommand, active: usize, seq: u64 },
    Fault(String),
}

#[derive(Clone)]
struct Request {
    seq: u64,
    left: f32,
    right: f32,
    facets: Vec<f32>,
}

/// Separate worker thread prevents Python + sparse matvec from blocking Vulkan.
pub struct NeuralLink {
    requests: Option<SyncSender<Request>>,
    events: Receiver<Event>,
    next_seq: u64,
    worker: Option<thread::JoinHandle<()>>,
}

fn python_program(root: &Path) -> PathBuf {
    if let Some(value) = env::var_os("FRUIT_FLY_PYTHON") { return PathBuf::from(value); }
    let windows = root.join(".venv").join("Scripts").join("python.exe");
    if windows.exists() { return windows; }
    let unix = root.join(".venv").join("bin").join("python");
    if unix.exists() { return unix; }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn spawn_python(root: &Path, mode: &str) -> std::io::Result<Child> {
    Command::new(python_program(root))
        .arg("-u").arg("-m").arg("fruit_fly_stuff.neural_bridge")
        .arg("--mode").arg(mode)
        .arg("--cache-dir").arg(root.join("data").join("fafb783").join("processed"))
        .arg("--checkpoint").arg(root.join("state").join(format!("bridge-{mode}.npz")))
        .current_dir(root)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn()
}

fn event_from_reply(reply: Reply) -> std::result::Result<Event, String> {
    match reply {
        Reply::Ready { version, mode, neurons, directed_pairs } => {
            if version != VERSION { return Err("incompatible neural bridge version".into()); }
            Ok(Event::Ready { mode, neurons, edges: directed_pairs })
        }
        Reply::Motor { version, seq, left, right, active } => {
            if version != VERSION || left > 1 || right > 1 {
                return Err("invalid neural motor response".into());
            }
            // Motion comes only from observed neuron spikes. The mapping
            // is synthetic/experimental, not a biological motor controller.
            Ok(Event::Motor {
                command: MotorCommand {
                    forward: f32::from(left + right) * 0.5,
                    turn: f32::from(right) - f32::from(left),
                },
                active, seq,
            })
        }
        Reply::Error { message } => Err(format!("neural worker rejected input: {message}")),
        Reply::Saved {} => Err("neural worker closed unexpectedly".into()),
    }
}

fn read_reply(lines: &mut impl BufRead) -> std::result::Result<Reply, String> {
    let mut line = String::new();
    let n = lines.read_line(&mut line).map_err(|e| e.to_string())?;
    if n == 0 { return Err("neural process closed its output".into()); }
    if n > 65536 { return Err("oversized neural bridge response".into()); }
    serde_json::from_str(&line).map_err(|e| format!("invalid neural JSON: {e}"))
}

impl NeuralLink {
    /// Set FRUIT_FLY_BRAIN_MODE=toy or full to enable. Disabled by default.
    pub fn from_environment() -> Option<Self> {
        let mode = match env::var("FRUIT_FLY_BRAIN_MODE") {
            Ok(value) if value == "toy" || value == "full" => value,
            Ok(value) if value != "off" && !value.is_empty() => {
                eprintln!("Unknown FRUIT_FLY_BRAIN_MODE={value:?} (expected toy/full/off)");
                return None;
            }
            _ => return None,
        };
        let root = env::var_os("FRUIT_FLY_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let (sender, requests) = mpsc::sync_channel::<Request>(1);
        let (events, receiver) = mpsc::channel::<Event>();
        let worker = thread::spawn(move || {
            let mut child = match spawn_python(&root, &mode) {
                Ok(child) => child,
                Err(error) => {
                    let _ = events.send(Event::Fault(format!("cannot launch Python: {error}")));
                    return;
                }
            };
            let Some(mut input) = child.stdin.take() else {
                let _ = events.send(Event::Fault("child stdin unavailable".into()));
                let _ = child.kill();
                return;
            };
            let Some(output) = child.stdout.take() else {
                let _ = events.send(Event::Fault("child stdout unavailable".into()));
                let _ = child.kill();
                return;
            };
            let mut output = BufReader::new(output);
            let startup = read_reply(&mut output)
                .and_then(|msg| match event_from_reply(msg)? {
                    ready @ Event::Ready { .. } => Ok(ready),
                    _ => Err("first bridge reply must be ready".into()),
                });
            match startup {
                Ok(ready) => { if events.send(ready).is_err() { let _ = child.kill(); return; } },
                Err(error) => {
                    let _ = events.send(Event::Fault(error));
                    let _ = child.kill();
                    let _ = child.wait();
                    return;
                }
            }
            for request in requests {
                let msg = Step {
                    kind: "step", version: VERSION, seq: request.seq,
                    left: request.left, right: request.right,
                    facets: request.facets,
                };
                let transaction = serde_json::to_writer(&mut input, &msg)
                    .map_err(|e| e.to_string())
                    .and_then(|_| input.write_all(b"\n").map_err(|e| e.to_string()))
                    .and_then(|_| input.flush().map_err(|e| e.to_string()))
                    .and_then(|_| read_reply(&mut output))
                    .and_then(event_from_reply);
                match transaction {
                    Ok(Event::Motor { seq, command, active }) if seq == request.seq => {
                        if events.send(Event::Motor { seq, command, active }).is_err() { break; }
                    }
                    Ok(_) => {
                        let _ = events.send(Event::Fault("out-of-order bridge response".into()));
                        break;
                    }
                    Err(error) => { let _ = events.send(Event::Fault(error)); break; }
                }
            }
            let _ = writeln!(input, "{{\"type\":\"close\",\"version\":1}}");
            let _ = input.flush();
            drop(input);
            let _ = child.wait();
        });
        Some(Self { requests: Some(sender), events: receiver, next_seq: 0, worker: Some(worker) })
    }

    /// Backpressure: drop extra control requests rather than queuing old sensory frames.
    pub fn send_senses(&mut self, left: f32, right: f32, facets: &[f32]) {
        let packet = Request {
            seq: self.next_seq, left: left.clamp(0.0, 1.0),
            right: right.clamp(0.0, 1.0),
            facets: facets.iter().map(|x| x.clamp(0.0, 1.0)).collect(),
        };
        if self.requests.as_ref().is_some_and(|tx| tx.try_send(packet).is_ok()) {
            self.next_seq += 1;
        }
    }

    pub fn poll(&self) -> Option<Event> {
        match self.events.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ready_and_spikes() {
        let ready = r#"{"type":"ready","version":1,"mode":"full","neurons":139255,"directed_pairs":15091983}"#;
        assert!(matches!(
            event_from_reply(serde_json::from_str(ready).unwrap()).unwrap(),
            Event::Ready { neurons: 139255, .. }
        ));
        let response = r#"{"type":"motor","version":1,"seq":7,"left":1,"right":0,"active":110}"#;
        match event_from_reply(serde_json::from_str(response).unwrap()).unwrap() {
            Event::Motor { command, seq: 7, .. } => {
                assert!(command.forward > 0.0 && command.turn < 0.0);
            }
            _ => panic!("expected motor"),
        };
    }

    #[test]
    fn malformed_motor_rejected() {
        let response = r#"{"type":"motor","version":1,"seq":1,"left":2,"right":0,"active":0}"#;
        assert!(event_from_reply(serde_json::from_str(response).unwrap()).is_err());
    }
}

impl Drop for NeuralLink {
    fn drop(&mut self) {
        // Closing the sender lets the worker send the Python "close" command
        // and wait for its checkpoint to hit disk before Rust exits.
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                eprintln!("Neural bridge worker panicked during shutdown");
            }
        }
    }
}
