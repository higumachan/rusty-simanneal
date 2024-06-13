use std::fmt::Debug;
use std::ops::{Neg, Sub};

use num_traits::{Num, Signed};
use rand::Rng;

use schedule::Schedule;

use crate::schedule::Progress;

mod schedule;

pub trait Transition<G: Rng>: Sized + Clone + Copy {
    type Context;

    fn choose(rng: &mut G, ctx: &Self::Context) -> Self;
}

pub trait AnnealingState<G: Rng>: Sized + Clone + Debug {
    type Energy: Ord + Clone + Copy + Debug + Sub<Output = Self> + Signed + Num + Neg + Into<f64>;
    type Transition: Transition<G, Context = Self::Context> + Debug;
    type Context;

    fn energy(&self) -> Self::Energy;
    fn apply(&mut self, ctx: &mut Self::Context, op: &Self::Transition) -> Option<()>;
}

pub trait AnnealingStateBack<G: Rng>: AnnealingState<G> {
    type Restore;

    fn apply_with_restore(
        &mut self,
        ctx: &mut Self::Context,
        op: &Self::Transition,
    ) -> Option<Self::Restore>;
    fn back(&mut self, ctx: &mut Self::Context, restore: &Self::Restore);
}

/// Simulated Annealing algorithm
/// minimize f(x) where x is a state
pub struct Annealer<G: Rng, S: AnnealingState<G>, C: Schedule> {
    pub state: S,
    pub ctx: S::Context,
    pub schedule: C,
}

impl<G: Rng, S: AnnealingState<G>, C: Schedule> Annealer<G, S, C> {
    pub fn new(state: S, ctx: S::Context, schedule: C) -> Self {
        Self {
            state,
            ctx,
            schedule,
        }
    }

    pub fn anneal(&mut self, rng: &mut G) -> S {
        let mut best_state = self.state.clone();
        let mut best_energy = self.state.energy();
        let mut current_energy = best_energy;
        let mut progress = Progress::zero();

        while self.schedule.should_continue(&progress) {
            let op = S::Transition::choose(rng, &self.ctx);
            if let Some(_restore) = self.state.apply(&mut self.ctx, &op) {
                let temperature = self.schedule.temperature(&progress);
                let new_energy = self.state.energy();
                let delta = (new_energy - current_energy).into();
                let p = rng.gen_range(0.0..=1.0);
                if delta.is_sign_positive() && (-delta / temperature).exp() > p {
                    self.state = self.state.clone();
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

impl<G: Rng, S: AnnealingStateBack<G>, C: Schedule> Annealer<G, S, C> {
    pub fn anneal_back(&mut self, rng: &mut G) -> S {
        let mut best_state = self.state.clone();
        let mut best_energy = self.state.energy();
        let mut current_energy = best_energy;
        let mut progress = Progress::zero();

        while self.schedule.should_continue(&progress) {
            let op = Transition::choose(rng, &self.ctx);
            if let Some(restore) = self.state.apply_with_restore(&mut self.ctx, &op) {
                let temperature = self.schedule.temperature(&progress);
                let new_energy = self.state.energy();
                let delta = (new_energy - current_energy).into();
                let p = rng.gen_range(0.0..=1.0);
                if delta.is_sign_positive() && (-delta / temperature).exp() > p {
                    self.state.back(&mut self.ctx, &restore);
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
