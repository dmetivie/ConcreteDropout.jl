using Markdown#hide

md"""
# Monte Carlo (Concrete) Dropout: Bayesian Neural Network (BNN) for a regression task
    
From [https://github.com/aurelio-amerio/ConcreteDropout/blob/main/examples/Tensorflow/regression_MCDropout.ipynb](https://github.com/aurelio-amerio/ConcreteDropout/blob/main/examples/Tensorflow/regression_MCDropout.ipynb).
For more information on this BNN implementation, see https://arxiv.org/pdf/1703.04977.pdf
"""

using Flux
using Flux.Optimise: update!
using Random
using StatsBase
using StatsPlots
using ConcreteDropoutLayer
using ConcreteDropoutLayer: Split

Random.seed!(MersenneTwister(1))

md"""
## Data
"""

function gen_data(N; in=1, out=1)
    X = randn(Float32, Q, N)
    σ = 1
    w = 2
    b = 8
    Y = w * ones(out, in) * X .+ b + σ * randn(Float32, out, N)
    return X, Y
end

md"""
## Neural network functions
"""

function build_model_concrete_dropout(in, out)
    DL_model = Chain(
        Dense(in => 100, relu),
        Dense(100 => 100),
        ConcreteDropout(),
        relu,
        Dense(100 => 100),
        ConcreteDropout(),
        relu
    )

    est_mean = Chain(Dense(100 => out), ConcreteDropout(), relu)
    est_logvar = Chain(Dense(100 => out), ConcreteDropout(), relu)

    return Chain(DL_model, Split(est_mean, est_logvar)) |> f32
end

function build_model_dropout(in, out, p)
    DL_model = Chain(
        Dense(in => 100, relu),
        Dense(100 => 100),
        Dropout(p),
        relu,
        Dense(100 => 100),
        Dropout(p),
        relu
    )

    est_mean = Chain(Dense(100 => out), Dropout(p), relu)
    est_logvar = Chain(Dense(100 => out), Dropout(p), relu)

    return Chain(DL_model, Split(est_mean, est_logvar)) |> f32
end

md"""
## Loss functions
"""

function heteroscedastic_loss(y_pred, y_true)
    D = size(y_pred, 1) ÷ 2
    μ = y_pred[1:D, :]
    log_var = y_pred[D+1:end, :]
    precision = exp.(-log_var)
    return sum(precision .* (y_true - μ) .^ 2 + log_var)
end

function full_loss(model, x, y; kwargs...)
    return heteroscedastic_loss(model(x), y) + add_CD_regularization(model[1]; kwargs...) + add_CD_regularization(model[2].paths[1]; kwargs...) + add_CD_regularization(model[2].paths[2]; kwargs...)
end

function simple_loss(model, x, y)
    return heteroscedastic_loss(model(x), y)
end

md"""
## Training functions
"""

function train_step!(model, opt_state, xy, func_loss)
    ## Calculate the gradient of the objective
    ## with respect to the parameters within the model:
    loss, grads = Flux.withgradient(model) do m
        func_loss(m, xy...)
    end
    update!(opt_state, model, grads[1])
    return loss
end

"""
    train!(model, opt, data, func_loss, x_test, y_test, epochs)
Train the `model` and comute at each epoch the training and testing loss
"""
function train!(model, opt, data, func_loss, x_test, y_test, epochs)
    v_loss = Float32[]
    t_loss = Float32[]
    loss = rand(Float32) # just to define loss in outer loop scope # probably better ways to do that
    for epoch in 1:epochs
        for d in data
            loss = train_step!(model, opt, d, func_loss)
        end
        test_loss = heteroscedastic_loss(model(x_test), y_test)
        append!(v_loss, loss)
        append!(t_loss, test_loss)
        @info "Epoch $epoch train_loss = $(round(loss, digits = 4)) test_loss = $(round(test_loss, digits = 4))"
    end
    return v_loss, model, opt, t_loss
end

md"""
# Data & Settings
"""

Q = 1
D = 1
n_train = 1000
n_test = 500

x_train, y_train = gen_data(n_train, in=Q, out=D)

x_test, y_test = gen_data(n_test, in=Q, out=D)

batch_size = 128
epochs = 100
data = Flux.DataLoader((x_train, y_train), batchsize=batch_size)

md"""
# Training
"""

md"""
## Dropout Model
"""

fix_dropout = 0.1
model_d = build_model_dropout(Q, D, fix_dropout)

md"""
Initialise the optimiser for this model:
"""
opt_state_d = Flux.setup(Adam(), model_d)
v_loss_d, model_out_d, opt_out_d, t_loss_d = train!(model_d, opt_state_d, data, simple_loss, x_test, y_test, epochs)

md"""
## Concrete Dropout Model
"""

md"""
Compute the regularisation values
"""
wr = get_weight_regularizer(n_train, l=1.0f-2, τ=1.0f0)
dr = get_dropout_regularizer(n_train, τ=1.0f0, cross_entropy_loss=false)

model = build_model_concrete_dropout(Q, D)
opt_state = Flux.setup(Adam(), model)
reg_loss(model, x, y) = full_loss(model, x, y; lw=wr, ld=dr)
v_loss, model_out, opt_out, t_loss = train!(model, opt_state, data, reg_loss, x_test, y_test, epochs)

md"""
# Result
"""

md"""
## Training loss
"""
begin
    p_train = plot(v_loss_d, label="Dropout($fix_dropout)", title="Train loss")
    plot!(v_loss, label="ConcreteDropout")
    xlabel!("Epoch")
    ylabel!("loss", yscale = :log10)
    p_test = plot(t_loss_d, label="test Dropout($fix_dropout)", title="Test loss")
    plot!(t_loss, label="test ConcreteDropout")
    xlabel!("Epoch")
    ylabel!("loss", yscale = :log10)
    plot(p_train, p_test)
end

md"""
## Monte Carlo predictions
"""

##! only work with D = out = 1 as it is coded now
"""
MC_predict(model, X::AbstractArray{T}; n_samples=1000, kwargs...)
For each X it returns `n_samples` monte carlo simulations where the randomness comes from the (Concrete)Dropout layers.
"""
function MC_predict(model, X::AbstractArray; n_samples=1000, heteroscedastic=true, kwargs...)
    Flux.testmode!(model, false) # (Concrete)Dropout will be active during predictions => Monte Carlo (Concrete)Dropout
    dim_out = Flux.outputsize(model, size(X))[1]
    D = heteroscedastic ? dim_out ÷ 2 : dim_out
    dim_N = ndims(X)
    mean_arr = zeros(D, size(X, dim_N))
    std_dev_arr = zeros(D, size(X, dim_N))

    X_in = similar(X, size(X)[1:end-1]..., n_samples)

    for (i, x) in enumerate(eachslice(X, dims=dim_N, drop = false))
        X_in[:] .= x
    
        predictions = model(X_in)
        θs_MC = predictions[1:D, :]
        logvars = predictions[D+1:end, :]

        θ_hat = mean(θs_MC, dims=2) # predictive_mean 

        θ2_hat = mean(θs_MC .^ 2, dims=2)
        var_mean = mean(exp.(logvars), dims=2) # aleatoric_uncertainty 
        total_var = θ2_hat - θ_hat .^ 2 + var_mean
        std_dev = sqrt.(total_var)

        mean_arr[:, i] .= θ_hat
        std_dev_arr[:, i] .= std_dev
    end
    Flux.testmode!(model) # turn back to default mode
    return mean_arr, std_dev_arr
end

y_pred, y_std = MC_predict(model_out, x_test)
y_pred_d, y_std_d = MC_predict(model_out_d, x_test)

md"""
Plot prediction plus standard deviation (aleatoric + epistemic)
"""
begin
    argsort = sortperm(x_test, dims=2)
    x_sorted = x_test[argsort]'
    y_true_sorted = y_test[argsort]'

    plot(x_sorted, y_true_sorted, label="y_test", lw = 2)
    plot!(x_sorted, y_pred_d[argsort]', ribbon=y_std_d[argsort]', label="ŷ ± σ Dropout(0.1)", alpha=0.4, lw = 1.5)
    plot!(x_sorted, y_pred[argsort]', ribbon=y_std[argsort]', label="ŷ ± σ ConcreteDropout", alpha=0.4, lw = 1.5)
end

md"""
Print all learned Dropout rates.
"""
for layer in [model_out[1]...; model_out[2][1]...; model_out[2][2]...]
    if layer isa ConcreteDropout
        println(layer)
    end
end
