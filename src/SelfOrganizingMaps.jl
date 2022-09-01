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
    σ²::Float64 = 0.05::(_ ≥ 0.0)
    topology::Symbol = :rectangular::(_ ∈ (:rectangular, :hexagonal, :spherical))
    η_decay::Symbol = :exponential::(_ ∈ (:asymptotic, :exponential))
    σ_decay::Symbol = :exponential::(_ ∈ (:asymptotic, :exponential, :none))
    neighbor_function::Symbol = :gaussian::(_ ∈ (:gaussian, :mexican_hat))
    matching_distance::PreMetric = euclidean
    Nepochs::Int = 1::(_ ≥ 1)
end


function MMI.fit(m::SelfOrganizingMap, verbosity::Int, X)
    Xmatrix = MMI.matrix(X)'  # make matrix p × n for efficiency

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

MMI.fitted_params(m::SelfOrganizingMap, fitresult) = (weights=fitresult.W, coords=fitresult.coords)



function MMI.transform(m::SelfOrganizingMap, fitresult, X)
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
    input = Union{AbstractMatrix{Continuous}, MMI.Table(Continuous)},
    output = MMI.Table(Continuous),
    path = "$(PKG).SelfOrganizingMap"
)


# NOTE: NEED TO ADD DOCUMENTATION

end
