use crate::{AnnealingStateBack, AnnealingStatePeeking, InitialState, Transition};

/// Semi auto test the implementation of AnnealingStatePeeking.
pub fn run_peeking_and_check<S: AnnealingStatePeeking>(
    rng: &mut impl rand::Rng,
    context: &S::Context,
    state: &mut S,
    check_num: usize,
    decimal_places: usize,
) {
    for _ in 0..check_num {
        let transition = S::Transition::choose(rng, context, state);
        let current_energy = state.energy(context);
        let new_energy = state.peek_energy(context, &transition, current_energy);
        if let Some(new_energy) = new_energy {
            state.apply(context, &transition);
            let new_energy_ref = state.energy(context);
            assert_eq!(
                round_decimal_places(new_energy.into(), decimal_places),
                round_decimal_places(new_energy_ref.into(), decimal_places)
            );
        }
    }
}

pub fn run_back_and_check<S: AnnealingStateBack>(
    rng: &mut impl rand::Rng,
    context: &S::Context,
    state: &mut S,
    check_num: usize,
    decimal_places: usize,
) {
    for _ in 0..check_num {
        let transition = S::Transition::choose(rng, context, state);
        let current_energy_ref = state.energy(context);
        let restore = state.apply_with_restore(context, &transition);
        if let Some(restore) = restore {
            let next_energy = state.energy(context);
            state.back(context, &restore);
            let current_energy = state.energy(context);
            assert_eq!(
                round_decimal_places(current_energy_ref.into(), decimal_places),
                round_decimal_places(current_energy.into(), decimal_places)
            );
            state.apply(context, &transition);
            let next_energy_ref = state.energy(context);
            assert_eq!(
                round_decimal_places(next_energy_ref.into(), decimal_places),
                round_decimal_places(next_energy.into(), decimal_places)
            );
        }
    }
}

// Round a floating point number to a specified number of decimal places.
fn round_decimal_places(value: f64, decimal_places: usize) -> f64 {
    format!("{:.1$}", value, decimal_places).parse().unwrap()
}
