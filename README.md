# Concrete Dropout in Julia

Implementation of the [Concrete Dropout layer](https://arxiv.org/pdf/1705.07832.pdf) by Y. Gal et al in Julia with the Deep Learning package [Flux.jl](https://fluxml.ai/).

The example [regression_MCDropout.ipynb](https://github.com/dmetivie/ConcreteDropout.jl/blob/main/example/regression_MCDropout.ipynb) showcase usage of Concrete Dropout layers in a context of Bayesian Neural Networks (see [this paper](https://arxiv.org/pdf/1703.04977.pdf)).

## Usage

### Download

Add this module (unregistered package) as

```julia
import Pkg
Pkg.add("https://github.com/dmetivie/ConcreteDropoutLayer.jl")
```

### Adding Concrete Dropout layer

Then add the layers like any other layers

```julia
using Flux # DeepLearning pkg
using ConcreteDropoutLayer

channel = 10

model = Chain(
        Conv((3,), channel => 64, relu),
        ConcreteDropout(; dims=(2, 3)), # ConcreteDropout Conv1D 
        Flux.MLUtils.flatten,
        Dense(6272 => 100, relu),
        ConcreteDropout(), # ConcreteDropout Dense
    )
```

```julia
X = rand(Float32, 100, channel, 258)
output = model(X)
```

If you want to use Concrete Dropout outside training, e.g. Monte Carlo Dropout, use `Flux.testmode!(model, false)`.

### Training

To add the regularisation to the loss as proposed in the [Concrete Dropout paper](https://arxiv.org/pdf/1705.07832.pdf) use

```julia
wr = get_weight_regularizer(n_train, l=1.0f-2, τ=1.0f0) # weight regularization hyperparameter
dr = get_dropout_regularizer(n_train, τ=1.0f0, cross_entropy_loss=false) # dropout hyperparameter

full_loss(model, x, y; kwargs...) = original_loss(model(x), y) + add_CD_regularization(model; kwargs...)
```

## TODO

- Clean regularization
  - Ideally the L2 part is directly in the optimizer with something like `OptimiserChain(WeightDecay(lw/(1-p)), Adam(0.1))`. And at each time step the value of `p` is `adjust!`. Or maybe with another normalization one could get rid of the `1/(1-p)`.
  - The entropy and L2 regularization are handled automatically i.e. all relevant layers (nested or not) are found quickly and adjust at every step.
- Implementation in [Lux.jl](https://lux.csail.mit.edu/) ?

## Acknowledgments

This code is inspired by the Python (tensorflow/pytorch) implementations of @aurelio-amerio, see [his module](https://github.com/aurelio-amerio/ConcreteDropout).
Thanks to @ToucheSir for some useful comments.
