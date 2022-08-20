module SelfOrganizingMaps

using Distances
using Random
using Tables

import MLJModelInterface
import MLJModelInterface: Continuous, Multiclass, metadata_model, metadata_pkg
import Random.GLOBAL_RNG


include("coordinates.jl")
include("SOM.jl")

export SOM
export train!
export SelfOrganizingMap

## CONSTANTS
const MMI = MLJModelInterface
const PKG = "SelfOrganizingMaps"

MMI.@mlj_model mutable struct SelfOrganizingMap <: MMI.Unsupervised
    k::Int = 10::(_ ≥ 2)
    η::Float64 = 0.5::(_ ≥ 0.0)
    σ²::Float64 = 1.0::(_ ≥ 0.0)
    topology::Symbol = :rectangular::(_ ∈ (:rectangular, :hexagonal, :spherical))
    η_decay::Symbol = :exponential::(_ ∈ (:asymptotic, :exponential))
    σ_decay::Symbol = :exponential::(_ ∈ (:asymptotic, :exponential, :none))
    neighbor_function::Symbol = :gaussian::(_ ∈ (:gaussian, :mexican_hat))
    matching_distance::PreMetric = euclidean
    Nepochs::Int = 1::(_ ≥ 1)
end


function MMI.fit(m::SelfOrganizingMap, verbosity::Int, X)
    schema = Tables.schema(X)
    Xmatrix = transpose(MMI.matrix(X))

    nfeatures = size(Xmatrix, 1)

    # 1. Build the SOM
    som = SOM(m.k,
              nfeatures,
              m.η,
              m.σ²,
              m.topology,
              m.η_decay,
              m.σ_decay,
              m.neighbor_function,
              m.matching_distance,
              m.Nepochs,
              )

    # 2. Train the SOM
    train!(som, Xmatrix)

    # 3. collect results
    cache  = nothing
    report = NamedTuple()

    return som, cache, report
end

MMI.fitted_params(m::SelfOrganizingMap, fitresult) = (weights=fitresult.W,)



function MMI.transform(m, fitresult, X)
    som = fitresult
    Xmatrix = transpose(MMI.matrix(X))

    node_labels = [Symbol("node_$(i)") for i ∈ 1:(m.k^2)]
    X̃ = pairwise(som.matching_distance, Xmatrix, som.W; dims=2)

    return MMI.table(X̃, names=node_labels, prototype=X)
end


metadata_pkg.(
    (SelfOrganizingMap,),
    name="SelfOrganizingMaps",
    uuid="ba4b7379-301a-4be0-bee6-171e4e152787",
    url="https://github.com/john-waczak/SelfOrganizingMaps.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
)

metadata_model(
    SelfOrganizingMap,
    human_name = "Self Organizing Map",
    input = MMI.Table(Continuous),
    #output = AbstractVector{Multiclass},
    output = MMI.Table(Continuous),
    weights = false,
    path = "$(PKG).SelfOrganizingMap"
)


# NOTE: NEED TO ADD DOCUMENTATION

end
