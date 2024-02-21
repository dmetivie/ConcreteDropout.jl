module ConcreteDropoutLayer

#! At present Flux/Functor does not allow trainable scalar, hence we cast p_logit at a 1-element vector.
using Flux
using Random

"""
```julia
mutable struct ConcreteDropout{F,D,R<:AbstractRNG}
  p_logit::F
  dims::D
  active::Union{Bool,Nothing}
  rng::R
end
```
`ConcreteDropout` layer as described in [Concrete Dropout paper](https://arxiv.org/pdf/1705.07832.pdf) by Y. Gal et al.
This layer applies a soft mask to the previous layer i.e. multiply by a number between 0 and 1 in selected `dims`.
Here is a reminder on typical `dims` setting depending on the type of previous layer
- On `Dense` layer, use `dims = :` # acts on all neurons and samples independently
- On "`Conv1D`", use `dims = (2,3)` # apply independently to each feature (channel) and all samples (but the same for the first dimension)
- On "`Conv2D`", use `dims = (3,4)` 
- On "`Conv3D`", use `dims = (4,5)`


Remember `Flux.jl` convention for `Dense` and `Conv` layers looks like `x_input = (DIM, F, N)`
  with `N` number of sample, `F` number of feature (channel) and DIM = nothing, LENGHT, (WIDTH, HEIGHT), (WIDTH, HEIGHT, DEPTH) depending respectively on the case Dense, 1D, 2D, 3D
  """
mutable struct ConcreteDropout{F,D,R<:AbstractRNG}
  p_logit::F
  dims::D
  active::Union{Bool,Nothing}
  rng::R
end
#TODO: In principle each ConcreteDropout layer has its own regularisation parameters. Hence they should be in the struct.

## Equivalent of forward (Pytorch) & call (tf)
(m::ConcreteDropout)(x) = concrete_dropout(m.rng, x, sigmoid(m.p_logit)[1] * Flux._isactive(m, x); dims=m.dims)

## Equivalent de __init__ (pytorch) & initialize/build (keras)
function ConcreteDropout(; T=Float32, init_min=T(0.1), init_max=T(0.1), active::Union{Bool,Nothing}=nothing, rng=Random.default_rng(), dims=:)
  init_min = log(init_min) - log(1 - init_min)
  init_max = log(init_max) - log(1 - init_max)

  p_logit = randn(rng, T, 1) * (init_max - init_min) .+ init_min
  return ConcreteDropout(p_logit, dims, active, rng)
end

"""
Relaxation term used as a soft mask in Concete Dropout (z in Eq. 5 of the [paper](https://arxiv.org/pdf/1705.07832.pdf) paper by Y. Gal et al):
```julia
z = 1 .- sigmoid.((log(p + ϵ) .- log1p(ϵ - p) .+ log.(x .+ ϵ) - log1p.(ϵ .- x))/ temperature)
```
"""
function compute_relaxation(x, p, ϵ, T)
  dr = (p + ϵ) / (1 + ϵ - p) * (x .+ ϵ) ./ (1 .- x .+ ϵ)
  return sigmoid(-log.(dr) / T)
end

"""
concrete_dropout(rng, x, p; ϵ=eps(eltype(x)), temperature=eltype(x)(0.1), dims=:)
Apply a ConcreteDropout mask to array `x` i.e. each `dims` of `x` is mutliply by a random number in [0,1].
"""
function concrete_dropout(rng, x, p; ϵ=eps(eltype(x)), temperature=eltype(x)(0.1), dims=:)
  ## this is the shape of the dropout mask, same as dropout  
  drop_prob = similar_dropout(x, dims)
  rand!(rng, drop_prob)

  drop_prob = compute_relaxation(drop_prob, p, ϵ, temperature)
  normalize = 1 / (1 - p)

  xMask = x .* drop_prob .* normalize ## we multiply the input by the concrete dropout mask and normalize

  return xMask
end

Flux.@functor ConcreteDropout

Flux.trainmode!(m::ConcreteDropout, mode=true) = (m.active = isnothing(Flux._tidy_active(mode)) ? nothing : mode; m)

md"""
## Utilities
"""

similar_dropout(x, dims::Colon) = similar(x)
similar_dropout(x, dims) = similar(x, ntuple(d -> d in dims ? size(x, d) : 1, ndims(x)))
input_feature(layer::Dense) = size(layer.weight, 2)
input_feature(layer::Conv) = size(layer.weight, ndims(layer.weight) - 1)

function Base.show(io::IO, d::ConcreteDropout)
  print(io, "ConcreteDropout(", round(sigmoid(d.p_logit)[1], digits=4))
  d.dims != (:) && print(io, ", dims=", d.dims)
  d.active == nothing || print(io, ", active=", d.active)
  print(io, ")")
end

## Optional 

"""
Base.eps(::Type{Flux.NilNumber.Nil}) = 0
Type piracy of the `Base.eps` function. Needed for `Flux.outputsize` to work.
  """
Base.eps(::Type{Flux.NilNumber.Nil}) = 0

md"""
# Regularization
"""

pen_l2(x::AbstractArray) = sum(abs2, x)

"""
Entropy of Bernoulli random variable with proba p
"""
entropy_Bernoulli(p) = p * log(p) + (1 - p) * log1p(-p)

"""
Compute the (differentiable) regularization terms needed for layers where ConcreteDropout is applied.
This term is to be added to the loss
"""
function add_CD_regularization(model; lw=1.0f-6, ld=1.0f-5)
  reg = zero(lw)
  #TODO find a better way 
  for (l, cd_layer) in enumerate(model)
    if isa(cd_layer, ConcreteDropout)
      layer = model[l-1]
      #TODO should in principle check that previous layer has weights
      reg += add_CD_regularization(layer, cd_layer, lw, ld)
    end
  end
  return reg
end

function add_CD_regularization(layer, cd_layer::ConcreteDropout, lw, ld)
  p = sigmoid(cd_layer.p_logit)[1] #! 1-element vector version

  K = input_feature(layer) # number of feature/channel. With appropriate `dims` choice only features (and samples) are dropped (not HEIGH, LENGTH, DEPTH)

  weights_regularizer = lw * pen_l2(layer.weight) / (1 - p) # 

  dropout_regularizer = ld * K * entropy_Bernoulli(p)
  return weights_regularizer + dropout_regularizer
end

get_weight_regularizer(N; l=1.0f-2, τ=1.0f-1) = l^2 / (τ * N)

function get_dropout_regularizer(N; τ=1.0f-1, cross_entropy_loss=false)
  reg = 1 / (τ * N)
  if !cross_entropy_loss
    reg *= 2
  end
  return reg
end

export ConcreteDropout
export add_CD_regularization, get_weight_regularizer, get_dropout_regularizer

include("split_layer.jl")
end