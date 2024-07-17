"""
Create `ConcreteDropout` layer either for `Flux.jl` or `Lux.jl`. It uses package extansion i.e. the layer is loaded only when `using` either `Flux.jl` or `Lux.jl`.
"""
module ConcreteDropoutLayer

# # https://github.com/JuliaLang/Pkg.jl/blob/6e9c2ca9a7cadc35b7bb740bc3e0c9a6bac482a1/docs/src/creating-packages.md#behavior-of-extensions
# extFlux = Base.get_extension(@__MODULE__, :ConcreteDropoutLayerFluxExt)
# println("load Flux: ", !isnothing(extFlux))
# if !isnothing(extFlux)
#     println("loaded: ", extFlux)
    
#     Split = extFlux.Split
#     get_dropout_regularizer = extFlux.get_dropout_regularizer
#     get_weight_regularizer = extFlux.get_weight_regularizer
#     add_CD_regularization = extFlux.add_CD_regularization
#     ConcreteDropout = extFlux.ConcreteDropout
#     export add_CD_regularization, get_weight_regularizer, get_dropout_regularizer
# end
# println("momo ", @__MODULE__)
# extLux = Base.get_extension(@__MODULE__, :ConcreteDropoutLayerLuxExt)
# println("load Lux: ", !isnothing(extLux))

# if !isnothing(extLux)
#     println("loaded: ", extLux)
    
#     ConcreteDropout = extLux.ConcreteDropout
# end

get_weight_regularizer(N; l=1.0f-2, τ=1.0f-1) = l^2 / (τ * N)

function get_dropout_regularizer(N; τ=1.0f-1, cross_entropy_loss=false)
    reg = 1 / (τ * N)
    if !cross_entropy_loss
        reg *= 2
    end
    return reg
end

"""
Entropy of Bernoulli random variable with proba p
"""
entropy_Bernoulli(p) = p * log(p) + (1 - p) * log1p(-p)


export get_weight_regularizer, get_dropout_regularizer


include("../ext/ConcreteDropoutLayerLuxExt/ConcreteDropoutLayerLuxExt.jl")

export ConcreteDropout


end