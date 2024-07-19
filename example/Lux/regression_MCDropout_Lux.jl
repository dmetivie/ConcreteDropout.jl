using Markdown#hide

md"""
# Monte Carlo (Concrete) Dropout: Bayesian Neural Network (BNN) for a regression task
    
From [https://github.com/aurelio-amerio/ConcreteDropout/blob/main/examples/Tensorflow/regression_MCDropout.ipynb](https://github.com/aurelio-amerio/ConcreteDropout/blob/main/examples/Tensorflow/regression_MCDropout.ipynb).
For more information on this BNN implementation, see https://arxiv.org/pdf/1703.04977.pdf
"""

using Lux
using Optimisers, Zygote
using Random
using MLUtils: DataLoader
using StatsBase
using StatsPlots
using ConcreteDropoutLayer#v0.0.6

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
function build_model_dropout(in, out)
    Chain(
        Dense(in => 100, relu),
        Dense(100 => 100, relu),
        ConcreteDropout(),
        Dense(100 => 100, relu),
        ConcreteDropout(),
        Parallel(nothing, 
            Chain(Dense(100 => out, relu), ConcreteDropout()),
            Chain(Dense(100 => out, relu), ConcreteDropout())
        )
    )
end

function build_model_dropout(in, out, p)
    Chain(
        Dense(in => 100, relu),
        Dense(100 => 100, relu),
        Dropout(p),
        Dense(100 => 100, relu),
        Dropout(p),
        Parallel(nothing, 
            Chain(Dense(100 => out, relu), Dropout(p)),
            Chain(Dense(100 => out, relu), Dropout(p))
        )
    )
end

md"""
## Loss functions
"""

function heteroscedastic_loss(y_pred, y_true)
    μ, log_var = y_pred
    precision = exp.(-log_var)
    return sum(precision .* (y_true - μ) .^ 2 + log_var)
end

function compute_loss_heteroscedastic(model, ps, st, (x, y))
    ŷ, st = model(x, ps, st)
    return heteroscedastic_loss(ŷ, y), st, ()
end

md"""
Version with the added regularization suggested in the original paper. `(names_CD, names_W, input_features), λp, λW` are provided and constant during the training.
"""
function compute_loss_heteroscedastic_w_reg(model, ps, st, (x, y), (names_CD, names_W, input_features), λp, λW)
    ŷ, st = model(x, ps, st)
    drop_rates, W = get_regularization(ps, names_CD, names_W)

    return heteroscedastic_loss(ŷ, y) + computeCD_reg(drop_rates, W, input_features, λp, λW), st, ()
end

md"""
## Training functions
"""

function train_step(train_state, xy, compute_loss)
    ## Calculate the gradient of the objective
    ## with respect to the parameters within the model:
    x, y = xy
    
    gs, loss, _, train_state = Lux.Experimental.compute_gradients(
                AutoZygote(), compute_loss, (x, y), train_state
    )
    train_state = Lux.Experimental.apply_gradients(train_state, gs)

    return loss, train_state
end


"""
	train(model, epochs, dataset, dataset_val, compute_loss; learning_rate=0.001f0, dev = cpu_device())
Train the `model` and comute at each epoch the training and testing loss
"""
function train(model, epochs, dataset, dataset_val, compute_loss; learning_rate=0.001f0, dev = cpu_device())
    ## Set up models
    rng = Xoshiro(0)

    train_state = Lux.Experimental.TrainState(rng, model, Adam(learning_rate); transform_variables=dev)

    ps = train_state.parameters
    st = train_state.states
    model = train_state.model

    ## Validation Loss
    losses_train = Float32[]
    x_val, y_val = dataset_val |> dev
    losses_val = Float32[first(compute_loss(model, ps, st, (x_val, y_val)))]
    loss = rand(Float32) # just to define loss in outer loop scope # probably better ways to do that
    best_test_state = train_state

    ## Training loop
    for epoch in 1:epochs
        issave = false
        for xy in dataset
            xy = xy |> dev
            loss, train_state = train_step(train_state, xy, compute_loss)
        end
        ps = train_state.parameters
        st = train_state.states
        loss_val = first(compute_loss(model, ps, st, (x_val, y_val)))
        if loss_val < minimum(losses_val)
            best_test_state = train_state
            issave = true
        end
        append!(losses_train, loss)
        append!(losses_val, loss_val)
        @info "Epoch $epoch train_loss = $(round(loss, digits = 4)) validation_loss = $(round(loss_val, digits = 4)) $(issave ? "Best model so far" : " ")"
    end
    return best_test_state, losses_train, losses_val
end

md"""
# Data & Settings
"""

Q = 1
D = 1
n_train = 1000
n_test = 500

using LuxCUDA
dev = gpu_device()

x_train, y_train = gen_data(n_train, in=Q, out=D)

x_test, y_test = gen_data(n_test, in=Q, out=D)

batch_size = 128
epochs = 100
data_train = DataLoader((x_train, y_train), batchsize=batch_size) |> dev
data_test = (x_test, y_test) |> dev

md"""
# Training
"""

md"""
## Dropout Model
"""

fix_dropout = 0.1
model_D = build_model_dropout(Q, D, fix_dropout)

#--------------------

@time "Dropout model" model_state_out_D, loss_train_D, loss_val_D = train(model_D, epochs, data_train, data_test, compute_loss_heteroscedastic; dev = gpu_device())

md"""
## Concrete Dropout Model
"""

md"""
### Without regularization
"""

model_CD = build_model_dropout(Q, D)
#---------------------

@time "Concrete Dropout model" model_state_out_CD, loss_train_CD, loss_val_CD = train(model_CD, epochs, data_train, data_test, compute_loss_heteroscedastic; dev = gpu_device())

md"""
### With regularization
"""

wr = get_weight_regularizer(n_train, l=1.0f-2, τ=1.0f0)
dr = get_dropout_regularizer(n_train, τ=1.0f0, cross_entropy_loss=false)

#------------
p_cd, w_cd, KK = regularization_infos(Lux.Experimental.TrainState(Xoshiro(0), model_CD, Adam(0.1f0); transform_variables=dev))
@time "Concrete Dropout model reg" model_state_out_CD_reg, loss_train_CD_reg, loss_val_CD_reg = train(model_CD, epochs, data_train, data_test, (model, ps, st, xy) -> compute_loss_heteroscedastic_w_reg(model, ps, st, xy, (p_cd, w_cd, KK), dr, wr); dev = gpu_device())

md"""
# Result
"""

md"""
## Training loss
"""
begin
    p_train = plot(loss_train_D, label="Dropout($fix_dropout)", title="Train loss")
    plot!(loss_train_CD, label="ConcreteDropout")
    plot!(loss_train_CD_reg, label="ConcreteDropout + reg", alpha = 0.7)
    xlabel!("Epoch")
    ylabel!("loss", yscale = :log10)
    p_test = plot(loss_val_D, label="test Dropout($fix_dropout)", title="Test loss")
    plot!(loss_val_CD, label="test ConcreteDropout")
    plot!(loss_val_CD_reg, label="test ConcreteDropout + reg", alpha = 0.7)    
    xlabel!("Epoch")
    ylabel!("loss", yscale = :log10)
    plot(p_train, p_test)
end

md"""
## Monte Carlo predictions
"""

"""
	MC_predict(model, X::AbstractArray{T}; n_samples=1000, kwargs...)
For each X it returns `n_samples` monte carlo simulations where the randomness comes from the (Concrete)Dropout layers.
"""
function MC_predict(model_state, X::AbstractArray, n_samples=1000; dev = gpu_device(), dim_out = model_state.model[end].layers[1].out_dims)
    st = model_state.states
    ps = model_state.parameters
    model = model_state.model

    dim_N = ndims(X)
    mean_arr = similar(X, dim_out, size(X, dim_N))
    std_dev_arr = similar(X, dim_out, size(X, dim_N))

    X = X |> dev
    X_in = similar(X, size(X)[1:end-1]..., n_samples) |> dev
    

    for (i, x) in enumerate(eachslice(X, dims=dim_N, drop = false))
        X_in .= x 
        predictions, st = model(X_in, ps, st)
        θs_MC, logvars = predictions |> cpu_device()

        θ_hat = mean(θs_MC, dims=2) # predictive_mean 

        θ2_hat = mean(θs_MC .^ 2, dims=2) # θ2_hat = mean(θs_MC' * θs_MC, dims=2)
        var_mean = mean(exp.(logvars), dims=2) # aleatoric_uncertainty 
        total_var = θ2_hat - θ_hat .^ 2 + var_mean
        std_dev = sqrt.(total_var)

        mean_arr[:, i] .= θ_hat
        std_dev_arr[:, i] .= std_dev
    end

    return mean_arr, std_dev_arr
end

y_pred, y_std = MC_predict(model_state_out_CD_reg, x_test, dev = dev, dim_out = D)
y_pred_d, y_std_d = MC_predict(model_state_out_D, x_test, dev = dev, dim_out = D)

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
function print_p_CD(l, ps, st, name)
    if l isa ConcreteDropout
        println("p = $(sigmoid(ps.p_logit)) of $name")
    end
    return l, ps, st
end;
Lux.Experimental.@layer_map print_p_CD model_state_out_CD.model model_state_out_CD.parameters model_state_out_CD.parameters;