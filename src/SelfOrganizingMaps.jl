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
    grid_type::Symbol = :rectangular::(_ ∈ (:rectangular, :hexagonal, :spherical))
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
              m.grid_type,
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
    # put class labels in the report
    classes = getBMUidx(som, Xmatrix)
    report = (; :classes=>classes)

    return som, cache, report
end

MMI.fitted_params(m::SelfOrganizingMap, fitresult) = (weights=fitresult.W', coords=fitresult.coords')



function MMI.transform(m::SelfOrganizingMap, fitresult, X)
    # return the coordinates of the bmu for each instance
    som = fitresult
    Xmatrix = transpose(MMI.matrix(X))

    res = zeros(size(Xmatrix, 2), 2)
    for i ∈ 1:size(Xmatrix, 2)
        bmu = getBMUidx(som, Xmatrix[:,i])
        res[i, 1] = som.coords[1, bmu]
        res[i, 2] = som.coords[2, bmu]
    end

    # bmus = getBMUidx(som, Xmatrix)
    return res
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
    output = AbstractMatrix{Continuous},
    path = "$(PKG).SelfOrganizingMap"
)



# ------------ documentation ------------------------


const DOC_SOM = "[Kohonen's Self Organizing Map](https://ieeexplore.ieee.org/abstract/document/58325?casa_token=pGue0TD38nAAAAAA:kWFkvMJQKgYOTJjJx-_bRx8n_tnWEpau2QeoJ1gJt0IsywAuvkXYc0o5ezdc2mXfCzoEZUQXSQ)"*
    ", Proceedings of the IEEE; Kohonen, T.; (1990):"*
    "\"The self-organizing map\""


"""
$(MMI.doc_header(SelfOrganizingMap))

SelfOrganizingMaps implements $(DOC_SOM)

# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: an `AbstractMatrix` or `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=10`: Number of nodes along once side of SOM grid. There are `k²` total nodes.
- `η=0.5`: Learning rate. Scales adjust made to winning node and its neighbors during each round of training.
- `σ²=0.05`: The (squared) neighbor radius. Used to determine scale for neighbor node adjustments.
- `grid_type=:rectangular`  Node grid geometry. One of `(:rectangular, :hexagonal, :spherical)`.
- `η_decay=:exponential` Learning rate schedule function. One of `(:exponential, :asymptotic)`
- `σ_decay=:exponential` Neighbor radius schedule function. One of `(:exponential, :asymptotic, :none)`
- `neighbor_function=:gaussian` Kernel function used to make adjustment to neighbor weights. Scale is set by `σ²`. One of `(:gaussian, :mexican_hat)`.
- `matching_distance=euclidean` Distance function from `Distances.jl` used to determine winning node.
- `Nepochs=1` Number of times to repeat training on the shuffled dataset.

# Operations
- `transform(mach, Xnew)`: returns the coordinates of the winning SOM node for each instance of `Xnew`. For SOM of `grid_type` `:rectangular` and `:hexagonal`, these are cartesian coordinates. For `grid_type` `:spherical`, these are the latitude and longitude in radians.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `coords`: The coordinates of each of the SOM nodes (points in the domain of the map) with shape (k², 2)

- `weights`: Array of weight vectors for the SOM nodes (corresponding points in the map's
  range) of shape (k², input dimension)

# Report
The fields of `report(mach)` are:
- `classes`: the index of the winning node for each instance of the training data X interpreted as a class label

# Examples
```
using MLJ
som = @load SelfOrganizingMap pkg=SelfOrganizingMaps
model = som()
X, y = make_regression(50, 3) # synthetic data
mach = machine(model, X) |> fit!
X̃ = transform(mach, X)

rpt = report(mach)
classes = rpt.classes
```
"""
SelfOrganizingMap



end
