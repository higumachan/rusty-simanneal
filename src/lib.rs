use std::fmt::Debug;
use std::ops::{Neg, Sub};

use log::debug;
use num_traits::{Num, Signed};
use rand::Rng;

use schedule::Schedule;

use crate::metrics::Metrics;
use crate::schedule::Progress;

mod metrics;
pub mod schedule;
pub mod test_implementer;

/// Transition is a trait to be implemented when the state can be updated by a transition.
/// e.g.
///```rust
/// use rand::Rng;
/// use rusty_simanneal::Transition;
///
/// #[derive(Debug, Clone, Copy)]
/// enum QuadraticFunctionTransition {
///     Add(f64),
///     Mul(f64),
/// }
///
/// #[derive(Debug, Clone)]
/// struct QuadraticFunctionState {
///     x: f64,
/// }
///
/// impl Transition for QuadraticFunctionTransition {
///     type Context = QuadraticFunction;
///     type State = QuadraticFunctionState;
///
///     fn choose<G: Rng>(rng: &mut G, _ctx: &Self::Context, _state: &Self::State) -> Self {
///         match rng.gen_range(0..=1) {
///             0 => Self::Add(rng.gen_range(-10.0..=10.0)),
///             1 => Self::Mul(rng.gen_range(0.3..=1.1)),
///             _ => unreachable!(),
///         }
///     }
/// }
///
/// struct QuadraticFunction {
///     a: f64,
///     b: f64,
///     c: f64,
/// }
///
/// let func = QuadraticFunction {
///     a: 1.0,
///     b: 10.0,
///     c: 30.0,
/// };
///
/// let state = QuadraticFunctionState { x: 100.0 };
///
/// let op = QuadraticFunctionTransition::choose(&mut rand::thread_rng(), &func, &state);
/// assert!(matches!(
///     op,
///     QuadraticFunctionTransition::Add(_) | QuadraticFunctionTransition::Mul(_)
/// ));
/// ```
pub trait Transition: Sized + Clone + Copy {
    type Context;
    type State;

    fn choose<G: Rng>(rng: &mut G, ctx: &Self::Context, state: &Self::State) -> Self;
}

/// EnergyMeasurable is a trait to be implemented when the energy of the state can be calculated.
/// e.g. quadratic function
/// ```rust
/// use rand::Rng;
/// use rusty_simanneal::{EnergyMeasurable, Transition};
///
/// struct QuadraticFunction {
///     a: f64,
///     b: f64,
///     c: f64,
/// }
/// #[derive(Debug, Clone)]
/// struct QuadraticFunctionState {
///     x: f64,
/// }
///
/// impl EnergyMeasurable for QuadraticFunctionState {
///     type Energy = f64;
///     type Context = QuadraticFunction;
///
///     fn energy(&self, ctx: &Self::Context) -> Self::Energy {
///         let f = ctx.a * self.x * self.x + ctx.b * self.x + ctx.c;
///         f * ctx.a.signum()
///     }
/// }
///
/// let func = QuadraticFunction {
///     a: 1.0,
///     b: 10.0,
///     c: 30.0,
/// };
/// let state = QuadraticFunctionState { x: -5.0 };
/// assert_eq!(state.energy(&func), 5.0);
/// ```
pub trait EnergyMeasurable: Sized + Clone + Debug {
    type Energy: PartialOrd + Clone + Copy + Debug + Sub + Signed + Num + Neg + Into<f64>;
    type Context;

    fn energy(&self, ctx: &Self::Context) -> Self::Energy;
}

/// InitialState is a trait to be implemented when the initial state can be generated.
/// e.g.
/// ```rust
/// use rand::Rng;
/// use rusty_simanneal::InitialState;
///
/// struct QuadraticFunction {
///    a: f64,
///    b: f64,
///    c: f64,
/// }
///
/// #[derive(Debug, Clone)]
/// struct QuadraticFunctionState {
///     x: f64,
/// }
///
/// impl InitialState for QuadraticFunction {
///     type Context = QuadraticFunctionState;
///     fn initial_state<G: Rng>(&self, rng: &mut G, _ctx: &Self::Context) -> QuadraticFunctionState {
///         QuadraticFunctionState {
///             x: rng.gen_range(-100.0..=100.0),
///        }
///    }
/// }
/// ```
pub trait InitialState {
    type Context;

    fn initial_state<G: Rng>(&self, rng: &mut G, ctx: &Self::Context) -> Self;
}

/// AnnealingState is a trait to be implemented when the state can be updated by a transition.
/// e.g. quadratic function
/// ```rust
/// use rand::prelude::ThreadRng;
/// use rand::Rng;
/// use rusty_simanneal::{Annealer, AnnealingState, EnergyMeasurable, schedule, Transition};
///
/// #[derive(Debug, Clone, Copy)]
/// enum QuadraticFunctionTransition {
///     Add(f64),
///     Mul(f64),
/// }
///
/// impl Transition for QuadraticFunctionTransition {
///     type Context = QuadraticFunction;
///     type State = QuadraticFunctionState;
///
///     fn choose<G: Rng>(rng: &mut G, _ctx: &Self::Context, _state: &Self::State) -> Self {
///         match rng.gen_range(0..=1) {
///             0 => Self::Add(rng.gen_range(-10.0..=10.0)),
///             1 => Self::Mul(rng.gen_range(0.3..=1.1)),
///             _ => unreachable!(),
///         }
///     }
/// }
///
/// struct QuadraticFunction {
///     a: f64,
///     b: f64,
///     c: f64,
/// }
/// #[derive(Debug, Clone)]
/// struct QuadraticFunctionState {
///     x: f64,
/// }
///
/// impl EnergyMeasurable for QuadraticFunctionState {
///     type Energy = f64;
///     type Context = QuadraticFunction;
///
///     fn energy(&self, ctx: &Self::Context) -> Self::Energy {
///         let f = ctx.a * self.x * self.x + ctx.b * self.x + ctx.c;
///         f * ctx.a.signum()
///     }
/// }
///
/// impl AnnealingState for QuadraticFunctionState {
///     type Transition = QuadraticFunctionTransition;
///
///     fn apply(&mut self, _ctx: &Self::Context, op: &Self::Transition) -> Option<()> {
///         match op {
///             QuadraticFunctionTransition::Add(dx) => {
///                 self.x += dx;
///             }
///             QuadraticFunctionTransition::Mul(rx) => {
///                 self.x *= rx;
///             }
///         }
///         Some(())
///     }
/// }
///
/// let func = QuadraticFunction {
///     a: 1.0,
///     b: 10.0,
///     c: 30.0,
/// };
///
/// let mut state = QuadraticFunctionState { x: 100.0 };
/// let mut annealer = Annealer::new(state, func, schedule::LinearStepSchedule::new(1000.0, 0.01, 10000));
/// let best_state = annealer.anneal::<_, false>(&mut rand::thread_rng());
/// assert!((best_state.x - (-5.0)).abs() < 0.1);
/// ```
pub trait AnnealingState: EnergyMeasurable {
    type Transition: Transition<Context = Self::Context, State = Self> + Debug;

    fn apply(&mut self, ctx: &Self::Context, op: &Self::Transition) -> Option<()>;
}

/// AnnealingStatePeeking is a trait to be implemented when the energy of the next state can be calculated efficiently without updating the state.
pub trait AnnealingStatePeeking: AnnealingState {
    /// Peek the energy of the state after applying the transition
    fn peek_energy(
        &self,
        ctx: &Self::Context,
        op: &Self::Transition,
        current_energy: Self::Energy,
    ) -> Option<Self::Energy>;
}

/// AnnealingStateBack is implemented when the state can be back processed more efficiently than clone.
pub trait AnnealingStateBack: AnnealingState {
    type Restore;

    /// Apply the transition and restore information for returning to the previous state
    fn apply_with_restore(
        &mut self,
        ctx: &Self::Context,
        op: &Self::Transition,
    ) -> Option<Self::Restore>;

    /// Restore the state to the previous state
    fn back(&mut self, ctx: &Self::Context, restore: &Self::Restore);
}

/// Simulated Annealing algorithm
/// minimize f(x) where x is a state
pub struct Annealer<S: EnergyMeasurable, C: Schedule> {
    pub state: S,
    pub ctx: S::Context,
    pub schedule: C,
    pub metrics: Vec<Metrics>,
}

impl<S: AnnealingState, C: Schedule> Annealer<S, C> {
    pub fn new(state: S, ctx: S::Context, schedule: C) -> Self {
        Self {
            state,
            ctx,
            schedule,
            metrics: Vec::new(),
        }
    }

    pub fn anneal<G: Rng, const METRICS: bool>(&mut self, rng: &mut G) -> S {
        let mut best_state = self.state.clone();
        let mut best_energy = self.state.energy(&self.ctx);
        let mut current_energy = best_energy;
        let mut progress = Progress::zero();

        if METRICS {
            self.metrics.clear();
        }

        while self.schedule.should_continue(&progress) {
            let start = if METRICS {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let prev_state = self.state.clone();
            let op = S::Transition::choose(rng, &self.ctx, &self.state);

            let (accept, improvement) = if let Some(_restore) = self.state.apply(&self.ctx, &op) {
                let temperature = self.schedule.temperature(&progress);
                let new_energy = self.state.energy(&self.ctx);

                let improvement = if new_energy < best_energy {
                    best_energy = new_energy;
                    best_state = self.state.clone();
                    true
                } else {
                    false
                };

                let delta = (new_energy - current_energy).into();
                let p = rng.gen_range(0.0..=1.0);
                if delta.is_sign_positive() && (-delta / temperature).exp() < p {
                    // reject
                    debug!("reject {} -> {}", current_energy.into(), new_energy.into());
                    self.state = prev_state;
                    (false, improvement)
                } else {
                    // accept
                    debug!("accept {} -> {}", current_energy.into(), new_energy.into());
                    current_energy = new_energy;
                    (true, improvement)
                }
            } else {
                (false, false)
            };

            if METRICS {
                self.metrics.push(Metrics {
                    best_energy: best_energy.into(),
                    current_energy: current_energy.into(),
                    next_energy: self.state.energy(&self.ctx).into(),
                    delta: (self.state.energy(&self.ctx) - current_energy).into(),
                    accept,
                    improvement,
                    progress: self.schedule.progress_0_1(&progress),
                    temperature: self.schedule.temperature(&progress),
                    step_duration: start.expect("METRICS = true").elapsed(),
                });
            }

            progress.update();
        }

        best_state
    }
}

impl<S: AnnealingStateBack, C: Schedule> Annealer<S, C> {
    /// Simulated Annealing algorithm
    /// minimize f(x) where x is a state
    /// Use BACK instead of CLONE when you want to abort and return to the state.
    pub fn anneal_back<G: Rng, const METRICS: bool>(&mut self, rng: &mut G) -> S {
        let mut best_state = self.state.clone();
        let mut best_energy = self.state.energy(&self.ctx);
        let mut current_energy = best_energy;
        let mut progress = Progress::zero();

        while self.schedule.should_continue(&progress) {
            let op = Transition::choose(rng, &self.ctx, &self.state);
            if let Some(restore) = self.state.apply_with_restore(&self.ctx, &op) {
                let temperature = self.schedule.temperature(&progress);
                let new_energy = self.state.energy(&self.ctx);
                let delta = (new_energy - current_energy).into();
                let p = rng.gen_range(0.0..=1.0);
                if delta.is_sign_positive() && (-delta / temperature).exp() < p {
                    self.state.back(&self.ctx, &restore);
                } else {
                    current_energy = new_energy;
                    if current_energy < best_energy {
                        best_energy = current_energy;
                        best_state = self.state.clone();
                    }
                }
            }
            progress.update();
        }

        best_state
    }
}

impl<S: AnnealingStatePeeking, C: Schedule> Annealer<S, C> {
    /// Simulated Annealing algorithm
    /// minimize f(x) where x is a state
    /// Use peek_energy instead of apply when the energy of the next state can be calculated efficiently without updating the state.
    pub fn anneal_peek<G: Rng, const METRICS: bool>(&mut self, rng: &mut G) -> S {
        let mut best_state = self.state.clone();
        let mut best_energy = self.state.energy(&self.ctx);
        let mut current_energy = best_energy;
        let mut progress = Progress::zero();

        while self.schedule.should_continue(&progress) {
            let op = Transition::choose(rng, &self.ctx, &self.state);
            if let Some(new_energy) = self.state.peek_energy(&self.ctx, &op, current_energy) {
                let temperature = self.schedule.temperature(&progress);
                let delta = (new_energy - current_energy).into();
                let p = rng.gen_range(0.0..=1.0);
                if !(delta.is_sign_positive() && (-delta / temperature).exp() < p) {
                    // accept
                    self.state.apply(&self.ctx, &op);
                    // assert_ulps_eq!(
                    //     new_energy.into(),
                    //     self.state.energy(&self.ctx).into(),
                    //     epsilon = 0.01
                    // );
                    current_energy = new_energy;
                    if current_energy < best_energy {
                        best_energy = current_energy;
                        best_state = self.state.clone();
                    }
                }
            }
            progress.update();
        }

        best_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// solve f(x) = a x^2 + b x + c
    #[derive(Debug, Clone)]
    struct QuadraticFunction {
        a: f64,
        b: f64,
        c: f64,
    }

    #[derive(Debug, Clone)]
    struct QuadraticFunctionState {
        x: f64,
    }

    #[derive(Debug, Clone, Copy)]
    enum QuadraticFunctionTransition {
        Add(f64),
        Mul(f64),
    }

    impl Transition for QuadraticFunctionTransition {
        type Context = QuadraticFunction;
        type State = QuadraticFunctionState;

        fn choose<G: Rng>(rng: &mut G, _ctx: &Self::Context, _state: &Self::State) -> Self {
            match rng.gen_range(0..=1) {
                0 => Self::Add(rng.gen_range(-10.0..=10.0)),
                1 => Self::Mul(rng.gen_range(0.3..=1.1)),
                _ => unreachable!(),
            }
        }
    }

    impl EnergyMeasurable for QuadraticFunctionState {
        type Energy = f64;
        type Context = QuadraticFunction;

        fn energy(&self, ctx: &Self::Context) -> Self::Energy {
            let f = ctx.a * self.x * self.x + ctx.b * self.x + ctx.c;
            f * ctx.a.signum()
        }
    }

    impl AnnealingState for QuadraticFunctionState {
        type Transition = QuadraticFunctionTransition;

        fn apply(&mut self, _ctx: &Self::Context, op: &Self::Transition) -> Option<()> {
            match op {
                QuadraticFunctionTransition::Add(x) => {
                    self.x += x;
                    Some(())
                }
                QuadraticFunctionTransition::Mul(x) => {
                    self.x *= x;
                    Some(())
                }
            }
        }
    }

    impl AnnealingStatePeeking for QuadraticFunctionState {
        fn peek_energy(
            &self,
            ctx: &Self::Context,
            op: &Self::Transition,
            _current_energy: Self::Energy,
        ) -> Option<Self::Energy> {
            let new_x = match op {
                QuadraticFunctionTransition::Add(x) => self.x + x,
                QuadraticFunctionTransition::Mul(x) => self.x * x,
            };
            let f = ctx.a * new_x * new_x + ctx.b * new_x + ctx.c;
            Some(f * ctx.a.signum())
        }
    }

    impl AnnealingStateBack for QuadraticFunctionState {
        type Restore = f64;

        fn apply_with_restore(
            &mut self,
            _ctx: &Self::Context,
            op: &Self::Transition,
        ) -> Option<Self::Restore> {
            let prev_x = self.x;
            match op {
                QuadraticFunctionTransition::Add(x) => {
                    self.x += x;
                }
                QuadraticFunctionTransition::Mul(x) => {
                    self.x *= x;
                }
            }
            Some(prev_x)
        }

        fn back(&mut self, _ctx: &Self::Context, restore: &Self::Restore) {
            self.x = *restore;
        }
    }

    #[test]
    fn solve_quadratic_function() {
        let mut annealer = Annealer::new(
            QuadraticFunctionState { x: 100.0 },
            QuadraticFunction {
                a: 1.0,
                b: 10.0,
                c: 30.0,
            },
            schedule::LinearStepSchedule::new(1000.0, 0.01, 10000),
        );

        let state = annealer.anneal::<_, false>(&mut rand::thread_rng());

        let QuadraticFunction { a, b, .. } = annealer.ctx;
        let answer = -b / (2.0 * a);

        dbg!(&state, answer, state.energy(&annealer.ctx));
        assert!((state.x - answer).abs() < 0.1);
    }

    #[test]
    fn solve_with_metrics() {
        let mut annealer = Annealer::new(
            QuadraticFunctionState { x: 100.0 },
            QuadraticFunction {
                a: 1.0,
                b: 10.0,
                c: 30.0,
            },
            schedule::LinearStepSchedule::new(1000.0, 0.01, 10000),
        );

        let state = annealer.anneal::<_, true>(&mut rand::thread_rng());

        let QuadraticFunction { a, b, .. } = annealer.ctx;
        let answer = -b / (2.0 * a);

        dbg!(&state, answer, state.energy(&annealer.ctx));
        assert!((state.x - answer).abs() < 0.1);
        assert_ne!(annealer.metrics.len(), 0);
    }
}
