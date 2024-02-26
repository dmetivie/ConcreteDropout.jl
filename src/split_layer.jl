# https://fluxml.ai/Flux.jl/stable/models/advanced/
## custom split layer
"""
  Split{T}
Takes an input and apply it in all `path` in `Split`
Example 
```julia
chain_in = Chain(Dense(100 => 200))
out_1 = Chain(Dense(200 => 2))
out_2 = Chain(Dense(200 => 1))
full_model = Chain(chain_in, Split(out_1, out_2))
N = 500
x_in = rand(Float32, 100, N)
full_model(x_in) # returns a vector of Matrix of size (3,N)
```

By default the outputs of the Split layer are concatenate (vcat). 
Use the following redefinition to produce separate arrays as in done [Flux documentation](https://fluxml.ai/Flux.jl/stable/models/advanced/).
```julia
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
``` 
"""
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

# I added the vcat for convernience, this assume that outputs have the same 2nd dim.
(m::Split)(x::AbstractArray) = vcat(map(f -> f(x), m.paths)...) 
Base.getindex(Split, i) = Split.paths[i]
