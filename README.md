# Concrete Dropout in Julia

Implementation of the [Concrete Dropout layer](https://arxiv.org/pdf/1705.07832.pdf) by Y. Gal et al in Julia with the Deep Learning package [Flux.jl](https://fluxml.ai/).
This is inspired by the Python (tensorflow/pytorch) implementations of [Aurelio Amerio](https://github.com/aurelio-amerio/ConcreteDropout).

The example regression_MCDropout.ipynb

## TODO

- Clean regularization
    - Ideally the L2 part is directly in the optimizer with something like `OptimiserChain(WeightDecay(lw/(1-p)), Adam(0.1))`. And at each time step the value of `p` is `adjust!`. Or maybe with another normalization one could get rid of the `1/(1-p)`.
    - The entropy and L2 regularization are handled automitatically i.e. all relevant layers (nested or not) are found quickly and adjust at every steps.
- Implementation in [Lux.jl](https://lux.csail.mit.edu/) ?
