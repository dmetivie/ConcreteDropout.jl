# Concrete Dropout in Julia

Implementation of the [Concrete Dropout layer](https://arxiv.org/pdf/1705.07832.pdf) by Y. Gal et al. in Julia with the Deep Learning package [Flux.jl](https://fluxml.ai/).

The notebook example [regression_MCDropout.ipynb](https://github.com/dmetivie/ConcreteDropout.jl/blob/main/example/regression_MCDropout.ipynb) showcases the usage of Concrete Dropout layers in the context of Bayesian Neural Networks (see [this paper](https://arxiv.org/pdf/1703.04977.pdf)).

**Warning**: I try to use Package extansion to have a version for Flux or Lux depending on which you load. Unfortunately, it was not as easy as I thought e.g. [this PR](https://github.com/JuliaLang/Pkg.jl/pull/3552/files#diff-1af5f877eb4497fc1f22daf47044d0958aa02ab39cc6da8ef052624870d75d28) and a lot of related question on Discourse. I am not sure what I was aiming for is currently possible easily.

For Flux version, I believe version v0.0.0 should work.

## Usage

### Download

Add this module as any unregistered Julia package

```julia
import Pkg
Pkg.add(url="https://github.com/dmetivie/ConcreteDropoutLayer.jl") 
```

### Adding a Concrete Dropout layer

Then add the layers like any other layers

```julia
using Flux
using ConcreteDropoutLayer

channel = 10

model = Chain(
        Conv((3,), channel => 64, relu),
        ConcreteDropout(; dims=(2, 3)), # ConcreteDropout for Conv1D layer
        Flux.MLUtils.flatten,
        Dense(6272 => 100, relu),
        ConcreteDropout(), # ConcreteDropout for Dense layer
    )
```

```julia
X = rand(Float32, 100, channel, 258)
output = model(X)
```

If you want to use Concrete Dropout outside training, e.g., Monte Carlo Dropout, use `Flux.testmode!(model, false)`.

### Training

To add the regularization to the loss as proposed in the [Concrete Dropout paper](https://arxiv.org/pdf/1705.07832.pdf) use

```julia
wr = get_weight_regularizer(n_train, l=1.0f-2, τ=1.0f0) # weight regularization hyperparameter
dr = get_dropout_regularizer(n_train, τ=1.0f0, cross_entropy_loss=false) # dropout hyperparameter

full_loss(model, x, y; kwargs...) = original_loss(model(x), y) + add_CD_regularization(model; kwargs...)
```

### API

```julia
mutable struct ConcreteDropout{F,D,R<:AbstractRNG}
  p_logit::F # logit value of the dropout probability
  dims::D # dimension to which the Dropout is applied
  active::Union{Bool,Nothing} # weather dropout is active or not
  rng::R # rng used for the dropout
end
```

Here is a reminder of the typical `dims` setting depending on the type of previous layer

- On `Dense` layer, use `dims = :` i.e. it acts on all neurons and samples independently
- On "`Conv1D`", use `dims = (2,3)` i.e. it applies Concrete Dropout independently to each feature (channel) and all samples (but it is the same for the first dimension)
- On "`Conv2D`", use `dims = (3,4)`
- On "`Conv3D`", use `dims = (4,5)`

## TODO

- Clean regularization
  - Ideally, the L2 term should directly be in the optimizer with something like `OptimiserChain(WeightDecay(lw/(1-p)), Adam(0.1))`.
And at each time step, the value of `p` is `adjust!`. Or maybe with another normalization, one could get rid of the `1/(1-p)`.
  - The entropy and L2 regularization are handled automatically, i.e., all relevant layers (nested or not) are found quickly and adjusted at every step.
- Implementation in [Lux.jl](https://lux.csail.mit.edu/) ?

## Acknowledgments

This code is inspired by the Python (tensorflow/pytorch) implementations of [@aurelio-amerio](https://github.com/aurelio-amerio), see [his module](https://github.com/aurelio-amerio/ConcreteDropout).
Thanks to [@ToucheSir](https://github.com/ToucheSir) for some useful comments.
