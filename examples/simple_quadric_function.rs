use rand::Rng;
use rusty_simanneal::{schedule, Annealer, AnnealingState, EnergyMeasurable, Transition};

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

// solve f(x) = a x^2 + b x + c
#[derive(Debug, Clone)]
struct QuadraticFunction {
    a: f64,
    b: f64,
    c: f64,
}

#[derive(Debug, Clone, Copy)]
struct QuadraticFunctionState {
    x: f64,
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

fn main() {
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
}
