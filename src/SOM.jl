using DataFrames
using Random

"""
    sphericaldistance(ϕ₁,θ₁,ϕ₂,θ₂)

Compute the distance between two points on the unit sphere with coordinates (ϕ₁, θ₁), and (ϕ₂, θ₂) using the physics convention for ϕ the polar angle and θ the azimuth.
"""
function sphericaldistance(ϕ₁,θ₁,ϕ₂,θ₂)
    return acos(sin(θ₁)*sin(θ₂)*cos(ϕ₁-ϕ₂) + cos(θ₁)*cos(θ₂))
end



"""
    euclideandistance(i,j,m,n)

Compute the euclidean distance between grid points (i,j) and (m,n), i.e.
`d = (i-m)^2 + (j-n)^2`
"""
function euclideandistance(i,j,m,n)
    return (i-m)^2 + (j-n)^2
end







# we want a supertype of SOM
# and then we can have a subtype of SquareSOM and SphereSOM
# both have a matrix of weights and a distance function
# SquareSOM has gridtype
# SphereSOM has x,y coordinates
# We could also do a generalized SOM that has Coordinates.
# weights are either in a 1d or 2d grid. The last dimension will
# match the input shape.


abstract type SOM end

struct SquareSOM{M<:AbstractArray, F<:Function} <: SOM
    w::M
    dist_func::F  # for computing neighborhood to nodes to update
    η₀::Float64  # base update factor
    λ::Float64  # update factor decay rate
    σ₀²::Float64 # base radius
    β::Float64  # radius decay rate
    nepochs::Int
    wrap::Symbol
end


struct SphericalSOM{M<:AbstractArray, F<:Function} <: SOM
    w
    dist_func::F # for computing neighborhood of nodes to update
    η₀::Float64  # base update factor
    λ::Float64  # update factor decay rate
    σ₀²::Float64 # base radius
    β::Float64  # radius decay rate
    nepochs::Int
    x
    y
end


# NOTE: we can set up weights and coordinates as static arrays for increased speed. The weight matrix will need to be mutable though.
"""
    SquareSOM(nfeatures::Int, M::Int, N::Int, dist_func; η₀ = 0.1, # base λ = 0.1, σ₀² = 1.0, β = 0.1, nepochs = 25, wrap=:none)

Generate a Self Organizing Map with a square topology.
"""
function SquareSOM(nfeatures::Int,
                   M::Int,
                   N::Int;
                   dist_func=euclideandistance,
                   η₀ = 0.1, # base
                   λ = 0.1,
                   σ₀² = 1.0,
                   β = 0.1,
                   nepochs = 25,
                   wrap=:none)

    return SquareSOM(rand(nfeatures,M,N),
                     dist_func,
                     η₀,
                     λ,
                     σ₀²,
                     β,
                     nepochs,
                     wrap)
end


# for the spherical version
function getBMUidx(w::AbstractArray{Float64, 2}, x::AbstractVector)
    D² = sum((w .- x) .^ 2, dims=1)
    idx = argmin(D²)
    return idx[2]
end


# for the square version
function getBMUidx(w::AbstractArray{Float64, 3}, x::AbstractVector)
    D² = sum((w.-x), dims=1)
    idx = argmin(D²)
    i = idx[2]
    j = idx[3]
    return return i, j
end


"""
    getBMUidx(som::SOM, x::AbstractVector)

Given a self organizing map `som` and input vector `x`, return the indices of the best matching unit (BMU).
"""
function getBMUidx(som::SOM, x::AbstractVector)
    return getBMUidx(som.w, x)
end


"""
    updateWeights!(som::SquareSOM, x::AbstractVector)

Update the weights of the self organizing map `som` given the feature vector `x`.
"""
function updateWeights!(som::SquareSOM, x::AbstractVector, η::Float64, σ²::Float64)
    M = size(som.w, 2)
    N = size(som.w, 3)

    g,h = getBMUidx(som, x)

    for i ∈ 1:M, j ∈ 1:N
        d² = euclideandistance(g,h,i,j)
        f = exp(-d²/(2σ²)) # decrease update the further away you are
        som.w[:, i, j] += η .* f .* (x .- som.w[:, i, j])
   end
end




"""
    train!(som::SquareSOM, df::DataFrame)

Using the DataFrame `df`, train the weights of the `som::SquareSOM`.
"""
function train!(som::SquareSOM, df::DataFrame)
    X = Matrix(df)
    η=som.η₀
    σ²=som.σ₀²

    # training loop
    for epoch ∈ som.nepochs
        shuffle!(X)
        # loop through each datapoint
        for x ∈ eachrow(X)
            updateWeights!(som, x, η, σ²)
        end

        # update the η and σ²
        η = som.η₀ * exp(-epoch * som.λ)
        σ² = som.σ₀² * exp(-epoch * som.β)
    end
end
