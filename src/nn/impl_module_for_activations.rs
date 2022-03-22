use super::traits::Module;
use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

impl<F: DifferentiableFunction> CanUpdateWithTape for F {
    fn update_with_tape(&mut self, _: &GradientTape) {}
}

impl<F: DifferentiableFunction> Randomize for F {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
}

impl<F: DifferentiableFunction + Default, T: Tensor> Module<T> for F {
    type Output = T;
    fn forward(&self, input: T) -> Self::Output {
        apply::<T, Self>(input)
    }
}