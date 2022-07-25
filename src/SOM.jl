
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
                   N::Int,
                   dist_func;
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



function getBMUidx(w::AbstractArray{Float64, 2}, x::AbstractVector)
    D² = sum((w .- x) .^ 2, dims=1)
    idx = argmin(D²)
    return idx[2]
end


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
    updateWeights!(SOM, x, η, σ², bmu)

"""
function updateWeights!(SOM, x, η, σ², bmu)
    M = size(SOM)[1]
    N = size(SOM)[2]

    g,h = bmu

    # if radius is small, only update bmu
    # otherwise change in cells via neighborhood function
    if σ² < 1e-3
        SOM[g,h,:] += η .* (x .- SOM[g, h, :])
    else
        for i ∈ 1:M, j ∈ 1:N
            d² = (i-g)^2 + (j-h)^2
            f = exp(-d²/(2σ²))
            SOM[i,j,:] += η .* f .* (x .- SOM[i,j,:])
        end
    end
end



