use std::time::{Duration, Instant};

pub trait Schedule {
    type Progress: Progress;

    fn progress_0_1(&self, progress: &Self::Progress) -> f64;

    fn should_continue(&self, progress: &Self::Progress) -> bool;
    fn temperature(&self, progress: &Self::Progress) -> f64;
}

pub trait Progress {
    type Maximum;

    fn zero() -> Self;
    fn update(&mut self);
    fn progress(&self, maximum: Self::Maximum) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct Step(pub usize);

impl Progress for Step {
    type Maximum = usize;

    fn zero() -> Self {
        Self(0)
    }

    fn update(&mut self) {
        self.0 += 1;
    }

    fn progress(&self, maximum: Self::Maximum) -> f64 {
        self.0 as f64 / maximum as f64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Time {
    start: Instant,
    current: Instant,
}

impl Progress for Time {
    type Maximum = Duration;

    fn zero() -> Self {
        let current = Instant::now();
        Self {
            start: current,
            current,
        }
    }

    fn update(&mut self) {
        self.current = Instant::now();
    }

    fn progress(&self, maximum: Self::Maximum) -> f64 {
        let elapsed = self.current - self.start;
        let total = maximum.as_secs_f64();
        elapsed.as_secs_f64() / total
    }
}

pub struct LinearStepSchedule {
    pub t_max: f64,
    pub t_min: f64,
    pub max_steps: usize,
}

impl LinearStepSchedule {
    pub fn new(tmax: f64, tmin: f64, max_steps: usize) -> Self {
        Self {
            t_max: tmax,
            t_min: tmin,
            max_steps,
        }
    }
}

impl Schedule for LinearStepSchedule {
    type Progress = Step;

    fn progress_0_1(&self, progress: &Self::Progress) -> f64 {
        progress.progress(self.max_steps)
    }

    fn should_continue(&self, progress: &Self::Progress) -> bool {
        progress.0 < self.max_steps
    }

    fn temperature(&self, progress: &Self::Progress) -> f64 {
        let progress = progress.progress(self.max_steps);
        self.t_max - (self.t_max - self.t_min) * progress
    }
}

pub struct LinearTimeSchedule {
    pub t_max: f64,
    pub t_min: f64,
    pub max_time: Duration,
}

impl LinearTimeSchedule {
    pub fn new(tmax: f64, tmin: f64, max_time: Duration) -> Self {
        Self {
            t_max: tmax,
            t_min: tmin,
            max_time,
        }
    }
}

impl Schedule for LinearTimeSchedule {
    type Progress = Time;

    fn progress_0_1(&self, progress: &Self::Progress) -> f64 {
        progress.progress(self.max_time)
    }

    fn should_continue(&self, progress: &Self::Progress) -> bool {
        progress.current - progress.start < self.max_time
    }

    fn temperature(&self, progress: &Self::Progress) -> f64 {
        let progress = progress.progress(self.max_time);
        self.t_max - (self.t_max - self.t_min) * progress
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use super::*;

    #[test]
    fn linear_step_scheduler() {
        let scheduler = LinearStepSchedule::new(1.0, 0.0, 10);
        let mut progress = Step::zero();
        progress.update();
        progress.update();
        progress.update();
        progress.update();
        progress.update();

        assert!(scheduler.should_continue(&progress));
        assert_eq!(scheduler.temperature(&progress), 0.5);
    }

    #[test]
    fn linear_time_scheduler() {
        let scheduler = super::LinearTimeSchedule::new(1.0, 0.0, Duration::from_millis(100));
        let mut progress = super::Time {
            start: Instant::now(),
            current: Instant::now(),
        };
        sleep(Duration::from_millis(50));
        progress.update();

        assert!(scheduler.should_continue(&progress));
        assert!(
            0.40 < scheduler.temperature(&progress) && scheduler.temperature(&progress) < 0.60,
            "Temperature: {}",
            scheduler.temperature(&progress)
        );
    }
}
