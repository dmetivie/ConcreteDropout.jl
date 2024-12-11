# module ConcreteDropoutLayerLuxExt
using Lux
using Random
using LuxLib.Impl: dropout_fptype, dropout_shape
using LuxCore: replicate
using Lux.Training: TrainState
using Functors

"""
    ConcreteDropout(; dims=:, temperature=0.1f0, init_p=(0.1f0, 0.1f0))

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
Lux.@concrete struct ConcreteDropout{T} <: Lux.AbstractLuxLayer
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
  realfptype = dropout_fptype(x)
  y = rand!(rng, similar(x, realfptype, dropout_shape(x, dims)))
  y = _concretedropout_kernel.(y, p, ϵ, temperature) .* invp
  return y
end

"""
  _concretedropout_kernel(x, p, ϵ, temperature)
Relaxation term used as a soft mask in Concete Dropout (z in Eq. 5 of the [paper](https://arxiv.org/pdf/1705.07832.pdf) paper by Y. Gal et al):
"""
_concretedropout_kernel(x, p, ϵ, temperature) = 1 - sigmoid((log(p + ϵ) - log1p(ϵ - p) + log(x + ϵ) - log1p(ϵ - x)) / temperature)

#* # Regularization 

#* ## Get regularization terms #TODO: better way to do this? Integrate L2 term inside Optimizer?

"""
  get_CD_infos(model_state::TrainState)
  get_CD_infos(model, ps, st)
Return a tuple with 
- The `Functors.KeyPath` of all `ConcreteDropout` layers
- The `Functors.KeyPath` of all layers to which the `ConcreteDropout` layers is applied.
- The number of input feature of all these layers.
For now are supported only `Dense` and `Conv` layers. To use another layer just overload `ConcreteDropout.input_feature(layer::TypeOfLayer) = INTEGER`.
```julia
key_cd, key_layer, feature_layer = get_CD_infos(model_state)
Lux.Functors.getkeypath(model_state.parameters, key_cd[1]) # return p_logit parameter of the ConcreteDropout
``` 
WARNING: the code base uses under the hood a custom version of [`Lux.Experimental.layer_map`](https://github.com/LuxDL/Lux.jl/blob/1ea272a135ad1ab2f3acc2d570c462434da5c02e/src/contrib/map.jl#L59) (which is already experimental) hence report any bug.
"""
function get_CD_infos(model_state::TrainState)
  return get_CD_infos(model_state.model, model_state.parameters, model_state.states)
end

function get_CD_infos(model::AbstractLuxLayer)
  ps, st = Lux.setup(Random.Xoshiro(0), model)
  return get_CD_infos(model, ps, st)
end

function get_CD_infos(model, ps, st)
  CD_names, W_names, W_type = layer_map_with_previous(model, ps, st)

  return CD_names, W_names, input_feature.(W_type)
end

#TODO: could supress t_layer and t_prev if https://github.com/LuxDL/Lux.jl/issues/1068 is fixed, in that case `getkeypath(model, kp_layer)` gives the types.
#TODO: moreover the shape of ps.weigth might be enough to distiguish from Dense and Conv without needed type??

function get_key_type!(kp_cd, kp_layer, t_layer, l, ps, st, name, name_prev, t_prev)
  if l isa ConcreteDropout
    push!(kp_cd, name)
    push!(kp_layer, name_prev)
    push!(t_layer, t_prev)
  end
  return l, ps, st
end

function layer_map_with_previous!(kp_cd, kp_layer, t_layer, l, ps, st)
  kp_prev = KeyPath(1)
  t_prev = Dense(1 => 1)
  Lux.Functors.fmap_with_path(l, ps, st; walk=Lux.Experimental.LayerWalkWithPath(), exclude=Lux.Experimental.layer_map_leaf) do kp, layer, ps_, st_
    l__, ps__, st__ = get_key_type!(kp_cd, kp_layer, t_layer, layer, ps_, st_, kp, kp_prev, t_prev)
    kp_prev = kp
    t_prev = layer
    return l__, ps__, st__ # needed for the code not to error but useless here
  end
  return kp_cd, kp_layer, t_layer
end

function layer_map_with_previous(l, ps, st)
  kp_cd = KeyPath[]
  kp_layer = KeyPath[]
  t_layer = AbstractLuxLayer[]
  layer_map_with_previous!(kp_cd, kp_layer, t_layer, l, ps, st)
end

#* ## Compute actual reg
input_feature(W::AbstractArray) = ndims(W) == 2 ? size(W, 2) : size(W, ndims(W) - 1)
input_feature(layer::Dense) = layer.in_dims
input_feature(layer::Conv) = layer.in_chs

"""
  get_CD_rates(ps, key_cd)
Return the array of (size one vector) of Concrete Dropout rate. `ps` is the `@NamedTuple` of the model parameters and `key_cd` the keypath (array or no) 
"""
get_CD_rates(ps, key_cd::AbstractArray{<:KeyPath}) = Base.Fix1(Functors.getkeypath, ps).(key_cd) .|> Base.Fix2(getfield, :p_logit) .|> sigmoid

get_CD_weigths(ps, key_w::AbstractArray{<:KeyPath}) = Base.Fix1(Functors.getkeypath, ps).(key_w) .|> Base.Fix2(getfield, :weight)

"""
  computeCD_reg(p, W, K, λp, λW)
Add the regularization term 
  sum(λW * sum(abs2, W[i]) ./ (1 .- p[i]) + λp * K[i] * entropy_Bernoulli.(p[i]) for i in eachindex(p)) |> sum
"""
computeCD_reg(p, W, K, λp, λW) = sum(λW .* sum.(abs2, W)./(1 .- sum.(p)) + λp .* K .* entropy_Bernoulli.(sum.(p)))

export TrainState
export get_CD_infos, get_CD_rates, get_CD_weigths
export computeCD_reg
# end