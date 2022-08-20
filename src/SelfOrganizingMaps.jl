module SelfOrganizingMaps

using Distances
using Random

import MLJModelInterface
import Random.GLOBAL_RNG

const MMI = MLJModelInterface

include("coordinates.jl")
include("SOM.jl")

export SOM
export train!


# MMI.@mlj_model mutable struct SelfOrganizingMap <: MMI.Unsupervised
#     k::Int = 100::(_ ≥ 2)
#     η::Float64 = 0.5::(_ ≥ 0.0)
#     σ²::Float64 = 1.0::(_ ≥ 0.0)
#     shape::Symbol = :rectangular::(_ ∈ (:rectangular, :hexagonal, :spherical))
#     η_decay::Symbol = :asymptotic::(_ ∈ (:asymptotic, :exponential))
#     neighbor_function::Synmbol = :gaussian::(_ ∈ (:gaussian, :mexican_hat))
#     matching_function::Metric = euclidean
#     nepochs::Int = 1::(_ ≥ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end


# function MMI.fit(m::DecisionTreeClassifier, verbosity::Int, X, y)
#     schema = Tables.schema(X)
#     Xmatrix = transpose(MMI.matrix(X))

#     # 1. Build the SOM
#     # 2. Train the SOM

#     cache  = nothing
#     report = NamedTuple()
#     return fitresult, cache, report
# end




end
