use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use serde::Deserialize;

use rusty_simanneal::{
    AnnealingState, AnnealingStateBack, AnnealingStatePeeking, EnergyMeasurable, Transition,
};
use rusty_simanneal::schedule::LinearStepSchedule;

// from simanneal import Annealer
// class TravellingSalesmanProblem(Annealer):
//
//     """Test annealer with a travelling salesman problem.
//     """
//
//     # pass extra data (the distance matrix) into the constructor
//     def __init__(self, state, distance_matrix):
//         self.distance_matrix = distance_matrix
//         super(TravellingSalesmanProblem, self).__init__(state)  # important!
//
//     def move(self):
//         """Swaps two cities in the route."""
//         # no efficiency gain, just proof of concept
//         # demonstrates returning the delta energy (optional)
//         initial_energy = self.energy()
//
//         a = random.randint(0, len(self.state) - 1)
//         b = random.randint(0, len(self.state) - 1)
//         self.state[a], self.state[b] = self.state[b], self.state[a]
//
//         return self.energy() - initial_energy
//
//     def energy(self):
//         """Calculates the length of the route."""
//         e = 0
//         for i in range(len(self.state)):
//             e += self.distance_matrix[self.state[i-1]][self.state[i]]
//         return e

#[derive(Deserialize, Debug, Clone)]
struct Row {
    #[serde(rename = "Town")]
    town: String,
    #[serde(rename = "Longitude")]
    longitude: f64,
    #[serde(rename = "Latitude")]
    latitude: f64,
}

#[derive(Debug, Clone)]
struct TspContext {
    distance_matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct TspState {
    route: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct TspTransition {
    a: usize,
    b: usize,
}

impl EnergyMeasurable for TspState {
    type Energy = f64;
    type Context = TspContext;

    fn energy(&self, ctx: &Self::Context) -> f64 {
        let mut e = 0.0;
        for i in 1..self.route.len() {
            e += ctx.distance_matrix[self.route[i - 1]][self.route[i]];
        }
        e += ctx.distance_matrix[self.route[self.route.len() - 1]][self.route[0]];
        e
    }
}

impl AnnealingState for TspState {
    type Transition = TspTransition;

    fn apply(&mut self, _ctx: &Self::Context, op: &Self::Transition) -> Option<()> {
        self.route.swap(op.a, op.b);
        Some(())
    }
}

impl AnnealingStateBack for TspState {
    type Restore = Self::Transition;

    fn apply_with_restore(
        &mut self,
        _ctx: &Self::Context,
        op: &Self::Transition,
    ) -> Option<Self::Restore> {
        let restore = *op;
        self.route.swap(op.a, op.b);
        Some(restore)
    }

    fn back(&mut self, _ctx: &Self::Context, restore: &Self::Restore) {
        self.route.swap(restore.a, restore.b);
    }
}

impl AnnealingStatePeeking for TspState {
    fn peek_energy(
        &self,
        ctx: &Self::Context,
        op: &Self::Transition,
        current_energy: Self::Energy,
    ) -> Option<Self::Energy> {
        let mut e = current_energy;

        let a = i64::try_from(op.a).unwrap();
        let b = i64::try_from(op.b).unwrap();

        let (a, b) = if a < b { (a, b) } else { (b, a) };
        let (a, b) = if a == 0 && b == (self.route.len() as i64 - 1) {
            (b, a)
        } else {
            (a, b)
        };

        let prev_a = (a - 1).rem_euclid(self.route.len() as i64) as usize;
        let next_a = (a + 1).rem_euclid(self.route.len() as i64) as usize;
        let prev_b = (b - 1).rem_euclid(self.route.len() as i64) as usize;
        let next_b = (b + 1).rem_euclid(self.route.len() as i64) as usize;

        if (a - b).abs().min(b + self.route.len() as i64 - a).abs() > 1 {
            e -= ctx.distance_matrix[self.route[prev_a]][self.route[a as usize]];
            e -= ctx.distance_matrix[self.route[a as usize]][self.route[next_a]];
            e -= ctx.distance_matrix[self.route[prev_b]][self.route[b as usize]];
            e -= ctx.distance_matrix[self.route[b as usize]][self.route[next_b]];

            e += ctx.distance_matrix[self.route[prev_a]][self.route[b as usize]];
            e += ctx.distance_matrix[self.route[b as usize]][self.route[next_a]];
            e += ctx.distance_matrix[self.route[prev_b]][self.route[a as usize]];
            e += ctx.distance_matrix[self.route[a as usize]][self.route[next_b]];
        } else {
            e -= ctx.distance_matrix[self.route[prev_a]][self.route[a as usize]];
            e -= ctx.distance_matrix[self.route[b as usize]][self.route[next_b]];

            e += ctx.distance_matrix[self.route[prev_a]][self.route[b as usize]];
            e += ctx.distance_matrix[self.route[a as usize]][self.route[next_b]];
        }

        Some(e)
    }
}

impl Transition for TspTransition {
    type Context = TspContext;
    type State = TspState;

    fn choose<G: Rng>(rng: &mut G, ctx: &Self::Context, state: &Self::State) -> Self {
        let a = rng.gen_range(0..state.route.len());
        let b = rng.gen_range(0..state.route.len());
        Self { a, b }
    }
}

fn main() {
    let mut reader = include_bytes!("data/location.txt").to_vec();
    let rdr = csv::ReaderBuilder::new().from_reader(&reader[..]);

    let rows: Result<Vec<_>, _> = rdr.into_deserialize::<Row>().collect();
    let rows = rows.unwrap();

    let mut distance_matrix = vec![vec![0.0; rows.len()]; rows.len()];
    for i in 0..rows.len() {
        for j in 0..rows.len() {
            let x = rows[i].longitude - rows[j].longitude;
            let y = rows[i].latitude - rows[j].latitude;
            distance_matrix[i][j] = (x * x + y * y).sqrt();
        }
    }

    let ctx = TspContext { distance_matrix };
    let state = TspState {
        route: (0..rows.len()).collect::<Vec<_>>().try_into().unwrap(),
    };

    {
        println!("start normal(clone) annealing");
        let state = state.clone();
        let mut rng = SmallRng::seed_from_u64(0);
        let mut annealer = rusty_simanneal::Annealer::new(
            state,
            ctx.clone(),
            LinearStepSchedule::new(100.0, 0.01, 10_000_000),
        );

        let start = Instant::now();
        let best_state = annealer.anneal::<_, false>(&mut rng);

        println!("process time {}ms", start.elapsed().as_millis());
        println!("{:?}", best_state);
        println!("{:?}", best_state.energy(&ctx));
    }

    {
        println!("start state back annealing");
        let state = state.clone();
        let mut rng = SmallRng::seed_from_u64(0);
        let mut annealer = rusty_simanneal::Annealer::new(
            state,
            ctx.clone(),
            LinearStepSchedule::new(100.0, 0.01, 10_000_000),
        );

        let start = Instant::now();
        let best_state = annealer.anneal_back::<_, false>(&mut rng);

        println!("process time {}ms", start.elapsed().as_millis());
        println!("{:?}", best_state);
        println!("{:?}", best_state.energy(&ctx));
    }

    {
        println!("start state peek energy annealing");
        let state = state.clone();
        let mut rng = SmallRng::seed_from_u64(0);
        let mut annealer = rusty_simanneal::Annealer::new(
            state,
            ctx.clone(),
            LinearStepSchedule::new(100.0, 0.01, 10_000_000),
        );

        let start = Instant::now();
        let best_state = annealer.anneal_peek::<_, true>(&mut rng);

        println!("process time {}ms", start.elapsed().as_millis());
        println!("{:?}", best_state);
        println!("{:?}", best_state.energy(&ctx));
    }
}
