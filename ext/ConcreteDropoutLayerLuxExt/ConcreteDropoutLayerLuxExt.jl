module ConcreteDropoutLayerLuxExt

using ConcreteDropoutLayer
using Random
using Lux
using LuxLib: _dropout_fptype, _dropout_shape
using LuxCore: replicate

"""
    ConcreteDropout(; dims=:)

`ConcreteDropout` layer as described in [Concrete Dropout paper](https://arxiv.org/pdf/1705.07832.pdf) by Y. Gal et al.
This layer applies a soft mask to the previous layer i.e. multiply by a number between 0 and 1 in selected `dims`.

## Keyword Arguments

  - To apply concrete dropout along certain dimension(s), specify the `dims` keyword. e.g.
    `Dropout(p; dims = (3, 4))` will randomly zero out entire channels on WHCN input
    (also called 2D concrete dropout).

## Inputs

  - `x`: Must be an AbstractArray

## Returns

  - `x` with dropout mask applied if `training=Val(true)` else just `x`
  - State with updated `rng`

## States

  - `rng`: Pseudo Random Number Generator
  - `training`: Used to check if training/inference mode

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`Dropout`](@ref), [`AlphaDropout`](@ref), [`VariationalHiddenDropout`](@ref)
"""
Lux.@concrete struct ConcreteDropout{T} <: Lux.AbstractExplicitLayer
    temperature::T
    dims
    init_p
end

function ConcreteDropout(; dims=:, temperature=0.1f0, init_p=(0.1f0, 0.1f0))
    ConcreteDropout(temperature, dims, init_p)
end

function Lux.initialstates(rng::AbstractRNG, ::ConcreteDropout)
    randn(rng)
    return (rng=Lux.replicate(rng), training=Lux.Val(true))
end

function Lux.initialparameters(rng::AbstractRNG, layer::ConcreteDropout{T}) where {T}
    init_min, init_max = layer.init_p
    init_min = log(init_min) - log(1 - init_min)
    init_max = log(init_max) - log(1 - init_max)

    p_logit = randn(rng, T, 1) * (init_max - init_min) .+ init_min
    return (p_logit=p_logit,)
end

Lux.parameterlength(l::ConcreteDropout) = 1

Lux.statelength(::ConcreteDropout) = 2

function (d::ConcreteDropout)(x, ps, st::NamedTuple)
    p = sigmoid(ps.p_logit[1])
    y, _, rng = concrete_dropout(st.rng, x, p, st.training, 1 / (1 - p), eps(eltype(x)), d.temperature, d.dims)
    return y, merge(st, (; rng))
end

function Base.show(io::IO, d::Dropout)
    print(io, "ConcreteDropout(")
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end

"""
    concrete_dropout(rng::AbstractRNG, x, p, ::Val{training}, invp, dims)

Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [1].
With ConcreteDropout the Dropout rate is a training variable and no longer an hyperparameter.

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `mask`: Dropout Mask. If not used then it is constructed automatically
  - `p`: Probability of an element to be dropped out
  - `Val(training)`: If `true` then dropout is applied on `x` with probability `p` along
    `dims`. Else, `x` is returned
  - `Val(update_mask)`: If `true` then the mask is generated and used. Else, the `mask`
    provided is directly used
  - `invp`: Inverse of the probability
  - `dims`: Dimensions along which dropout is applied
  - `invp`: Inverse of the probability (``\frac{1}{1-p}``)

## Returns

  - Output Array after applying dropout
  - Dropout Mask (if `training == false`, the returned value is meaningless)
  - Updated state for the random number generator

## References

[1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from
    overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.
"""
function concrete_dropout(
    rng::AbstractRNG, x::AbstractArray, p::T, ::Val{true}, invp::T, ϵ, temperature, dims) where {T}
    rng = replicate(rng)
    mask = _generate_concretedropout_mask(rng, x, p, invp, ϵ, temperature; dims)
    return (x .* (mask), mask, rng)
end

function concrete_dropout(
    rng::AbstractRNG, x::AbstractArray, p::T, ::Val{false}, ::T, ϵ, temperature, dims) where {T}
    return (x, x, rng)
end

# function concrete_dropout(rng::AbstractRNG, x::AbstractArray, ::AbstractArray,
#         p::T, t::Val, ::Val{true}, invp::T, ϵ, temperature, dims) where {T}
#     return concrete_dropout(rng, x, p, t, invp, ϵ, temperature, dims)
# end

# function concrete_dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
#         p::T, ::Val{true}, ::Val{false}, invp::T, ϵ, temperature, dims) where {T, T1, T2, N}
#     size(x) != size(mask) && return concretedropout(rng, x, p, Val(true), invp, dims)
#     return x .* (mask), mask, rng
# end

# function concrete_dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
#         p::T, ::Val{false}, ::Val{false}, invp::T, ϵ, temperature, dims) where {T, T1, T2, N}
#     return (x, mask, rng)
# end

function _generate_concretedropout_mask(rng::AbstractRNG, x, p, invp, ϵ, temperature; dims)
    realfptype = _dropout_fptype(x)
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y = _concretedropout_kernel.(y, p, ϵ, temperature) * invp
    return y
end

"""
Relaxation term used as a soft mask in Concete Dropout (z in Eq. 5 of the [paper](https://arxiv.org/pdf/1705.07832.pdf) paper by Y. Gal et al):
"""
_concretedropout_kernel(x, p, ϵ, temperature) = 1 - sigmoid((log(p + ϵ) - log1p(ϵ - p) + log(x + ϵ) - log1p(ϵ - x)) / temperature)

end