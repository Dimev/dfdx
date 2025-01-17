/// Immutable forward of `Input` that produces [Module::Output].
/// See [ModuleMut] for mutable forward.
pub trait Module<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Forward `Input` through the module and produce [Module::Output].
    ///
    /// **See [ModuleMut] for version that can mutate `self`.**
    ///
    /// Example Usage:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward(Tensor2D::zeros());
    /// ```
    fn forward(&self, input: Input) -> Self::Output;
}

/// Mutable forward of `Input` that produces [ModuleMut::Output].
/// See [Module] for immutable forward.
pub trait ModuleMut<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Forward `Input` through the module and produce [ModuleMut::Output].
    ///
    /// **See [Module::forward()] for immutable version**
    ///
    /// Example Usage:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let mut model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward_mut(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward_mut(Tensor2D::zeros());
    /// ```
    fn forward_mut(&mut self, input: Input) -> Self::Output;
}

/// Something that can reset it's parameters.
pub trait ResetParams {
    /// Mutate the unit's parameters using [rand::Rng]. Each implementor
    /// of this trait decides how the parameters are initialized. In
    /// fact, some impls may not even use the `rng`.
    ///
    /// # Example:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// struct MyMulLayer {
    ///     scale: Tensor1D<5, NoneTape>,
    /// }
    ///
    /// impl ResetParams for MyMulLayer {
    ///     fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
    ///         for i in 0..5 {
    ///             self.scale.mut_data()[i] = rng.gen_range(0.0..1.0);
    ///         }
    ///     }
    /// }
    /// ```
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R);
}
