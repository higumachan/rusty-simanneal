use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Metrics {
    pub best_energy: f64,
    pub current_energy: f64,
    pub next_energy: f64,
    pub delta: f64,
    pub accept: bool,
    pub improvement: bool,
    pub progress: f64,
    pub temperature: f64,
    pub step_duration: Duration,
}
