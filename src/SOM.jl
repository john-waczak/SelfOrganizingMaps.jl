struct SOM{T<:AbstractArray, T2<:AbstractArray,  R<:AbstractFloat, F<:Function, F2<:Function, M<:PreMetric, M2<:PreMetric}
    W::T  # weigts n x d matrix
    coords::T2 # 2 x d  smatrix of lnode coordinates
    σ²::R  # radius for neighbor distance
    η::R  # learning rate
    η_decay::F  # [:exponential, :asymptotic]
    σ_decay::F  # [:exponential, :asymptotic]
    neighbor_function::F2 #[:gaussian, :mexican_hat]
    neighbor_distance::M  # any metric from Distances.jl i.e. [euclidean, spherical_angle, cityblock, etc...]
    matching_function::M2  # [euclidean, cosine_dist, etc...]
    Nepochs::Int
end


"""
    function exponential_decay(η, i, N)


"""
function exponential_decay(η, i, N)
    return η*exp(-i/N)
end

function asymptotic_decay(η, i, N)
    return η/(1 + i/(N/2))  # see MiniSOM.py
end


function gaussian_neighbor(d, σ²)
    return exp(-d^2/(2σ²))
end

function mexicanhat_neighbor(d, σ²)
    return (1- d^2/σ²)*exp(-d^2/(2σ²))
end


function SOM(k::Int,
             nfeatures::Int,
             η::Float64,
             σ²::Float64,
             topology::Symbol,
             η_decay::Symbol,
             σ_decay::Symbol,
             neighbor_function::Symbol,
             matching_function::PreMetric,
             Nepochs::Int
             )
    # generate W, the matrix of weights of shape nfeatures × k² nodes
    W = rand(Float32, (nfeatures, k^2))

    # generate a k×k grid for :rectangular or hexagonal, otherwise, generate k² points
    if topology == :hexagonal
        coords = gethexpoints(k)
        neighbor_distance = euclidean
    elseif topology == :spherical
        coords = getspherepoints(k^2)
        neighbor_distance = spherical_angle
    else
        coords = getsquarepoints(k)
        neighbor_distance = euclidean
    end

    # set the function for decreasing the learning rate
    if η_decay == :exponential
        decay_func = exponential_decay
    else
        decay_func = asymptotic_decay
    end

    # set the function for neighborhood updates
    if neighbor_function == :mexican_hat
        nfunc = mexicanhat_neighbor
    else
        nfunc = gaussian_neighbor
    end

    return SOM(W, coords,
               σ², η,
               decay_func, nfunc,
               neighbor_distance, matching_function,
               Nepochs
               )
end




# for the spherical version
function getBMUidx(som::SOM, x::AbstractVector)
    d = colwise(som.matching_function, som.W, rand(3))
    idx = argmin(d)
    return idx
end


function updateWeights!(som::SOM, x::AbstractVector, step::Int, Nsteps::Int)
    N = size(som.W, 2)
    idx = getBMUidx(som, x)

    for i ∈ 1:size(som.W, 2)
        # compute distance to bmu
        d = evaluate(som.neighbor_distance, som.coords[:,i], som.coords[:,idx])
        f = som.neighbor_function(d, som.σ²)
        som.W[:,i] += som.η_decay(som.η, step, Nsteps) .* f .* (x .- som.W[:,i])
    end

end



function train!(som::SOM, X::AbstractArray)
    # training loop
    Nsteps = size(X, 2)

    for epoch ∈ 1:som.Nepochs
        shuffle!(X)
        # loop through each datapoint
        for i ∈ 1:Nsteps
            updateWeights!(som, X[:,i], i, Nsteps)
        end
    end
end
