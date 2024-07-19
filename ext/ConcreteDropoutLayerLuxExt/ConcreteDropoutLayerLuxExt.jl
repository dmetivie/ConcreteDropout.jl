# module ConcreteDropoutLayerLuxExt
using Lux
using Random
using Lux.LuxLib: _dropout_fptype, _dropout_shape
using Lux.LuxCore: replicate

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
    p = sigmoid.(ps.p_logit)
    y, _, rng = concrete_dropout(st.rng, x, p, st.training, 1 ./ (1 .- p), eps(eltype(x)), d.temperature, d.dims)
    return y, merge(st, (; rng))
end

function Base.show(io::IO, d::ConcreteDropout)
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
    y = _concretedropout_kernel.(y, p, ϵ, temperature) .* invp
    return y
end

"""
Relaxation term used as a soft mask in Concete Dropout (z in Eq. 5 of the [paper](https://arxiv.org/pdf/1705.07832.pdf) paper by Y. Gal et al):
"""
_concretedropout_kernel(x, p, ϵ, temperature) = 1 - sigmoid((log(p + ϵ) - log1p(ϵ - p) + log(x + ϵ) - log1p(ϵ - x)) / temperature)

#* # Regularization 

#* ## Get regularization terms #TODO: better way to do this? Integrate L2 term inside Optimizer?

function get_regularization(model_state)
  p_cd, w_cd, K_cd = regularization_infos(model_state)
  return get_regularization(model_state.parameters, p_cd, w_cd)
end

function get_regularization(ps, p_cd, w_cd)
  rates = get_regularization(ps, p_cd)

  W = [getproperty(ps, w) for w in w_cd]

  return rates, W
end

function get_regularization(ps, p_cd)
  rates = [sigmoid.(getproperty(ps, p)) for p in p_cd] # to work on GPU p and rate have to be vector
  return rates
end

function get_regularization(model_state, freeze::Bool)
  p_cd = regularization_infos(model_state.model, model_state.parameters, model_state.states, freeze)
  return get_regularization(model_state.parameters, p_cd)
end

"""
  regularization_infos(model_state)
Brut force extract the name of all layers/parameters involved in ConcreteDropout. 

```julia
p_cd, w_cd = regularization_infos(model_state)
eval(Meta.parse(w_cd[1])) # return a weigth matrix
``` 
"""
function regularization_infos(model_state::Lux.Experimental.TrainState)
  return regularization_infos(model_state.model, model_state.parameters, model_state.states)
end

function regularization_infos(model_state::Lux.Experimental.TrainState, freeze::Bool)
  return regularization_infos(model_state.model, model_state.parameters, model_state.states, freeze)
end

function regularization_infos(model, ps, st)
  CD_layer_names = String[]
  function print_p_CD(l, ps, st, name)
      if l isa ConcreteDropout
          push!(CD_layer_names, name)
      end
      return l, ps, st
  end

  Lux.Experimental.@layer_map print_p_CD model ps st
  CD_ps_names = replace.(CD_layer_names, "model." => "")
  CD_ps_names = replace.(CD_ps_names, "layers." => "")

  CD_ps_nb = [parse(Int,a[end]) - 1 for a in CD_ps_names]
  W_ps_names = [string(a[1:end-1], CD_ps_nb[i]) for (i,a) in enumerate(CD_ps_names)]
  CD_ps_names = [(string(a, ".p_logit")) for a in CD_ps_names]
  W_ps_names = [(string(a, ".weight")) for a in W_ps_names]

  w_cd = Meta.parse.(W_ps_names)

  return Meta.parse.(CD_ps_names), w_cd, [input_feature(getproperty(ps, w)) for w in w_cd]
end

#TODO: this is very lazy multiple dispach for the case you don't care about weigths-> improve
function regularization_infos(model, ps, st, freeze)
  CD_layer_names = String[]
  function print_p_CD(l, ps, st, name)
      if l isa ConcreteDropout
          push!(CD_layer_names, name)
      end
      return l, ps, st
  end

  Lux.Experimental.@layer_map print_p_CD model ps st
  CD_ps_names = replace.(CD_layer_names, "model." => "")
  CD_ps_names = replace.(CD_ps_names, "layers." => "")

  CD_ps_names = [(string(a, ".p_logit")) for a in CD_ps_names]

  return Meta.parse.(CD_ps_names)
end

"""
    getproperty(object, nested_name::Expr)

Call getproperty recursively on `object` to extract the value of some
nested property, as in the following example:

    julia> object = (X = (x = 1, y = 2), Y = 3)
    julia> getproperty(object, :(X.y))
    2
[Code from Anthony Blaom on discourse](https://discourse.julialang.org/t/nested-getproperty-requests/27968)
"""
function Base.getproperty(obj, ex::Expr)
    subex, field = reduce_nested_field(ex)
    return getproperty(getproperty(obj, subex), field)
end

# applying the following to `:(a.b.c)` returns `(:(a.b), :c)`
function reduce_nested_field(ex)
    ex.head == :. || throw(ArgumentError)
    tail = ex.args[2]
    tail isa QuoteNode || throw(ArgumentError)
    field = tail.value
    field isa Symbol || throw(ArgmentError)
    subex = ex.args[1]
    return (subex, field)
end

#* ## Compute actual reg
input_feature(W::AbstractArray) = ndims(W) == 2 ? size(W, 2) : size(W, ndims(W) - 1)
input_feature(layer::Dense) = size(layer.weight, 2)
input_feature(layer::Conv) = size(layer.weight, ndims(layer.weight) - 1)

"""
  Add the regularization term 
"""
function computeCD_reg(p, W, K, λp, λW)
  sum(λW*sum(abs2, W[i])./(1 .- p[i]) + λp*K[i]*entropy_Bernoulli.(p[i]) for i in eachindex(p)) |> sum
end

function computeCD_reg(p, K, λp)
  sum(λp*K[i]*entropy_Bernoulli.(p[i]) for i in eachindex(p)) |> sum
end


export regularization_infos, getproperty, get_regularization
export computeCD_reg
# end