using SelfOrganizingMaps
using Test
using Distances
using MLJBase
using StableRNGs  # should add this for later
using MLJTestInterface

stable_rng() = StableRNGs.StableRNG(1234)

@testset "coordinates.jl" begin
    # test spherical points
    N = 100
    X = SelfOrganizingMaps.getspherepoints(N)
    @test size(X) == (2, N)
    @test isapprox(spherical_angle(X[:,1], X[:,2]), spherical_angle(X[:,2], X[:,3]), atol=0.01)  # the spherical points aren't perfectly equi-distributed

    # test square points
    k = 10
    X = SelfOrganizingMaps.getsquarepoints(k)
    @test size(X) == (2, k^2)
    @test euclidean(X[:,1], X[:,2]) ≈ euclidean(X[:,2], X[:,3])

    # test hex points
    k = 10
    X = SelfOrganizingMaps.gethexpoints(k)
    @test size(X) == (2, k^2)
    @test euclidean(X[:,1], X[:,2]) ≈ euclidean(X[:,2], X[:,3])
end


@testset "SOM.jl" begin
    # test decay functions
    @test SelfOrganizingMaps.exponential_decay(1, 1,1) == 1.0
    @test SelfOrganizingMaps.asymptotic_decay(1, 1, 1) == 1.0
    @test SelfOrganizingMaps.no_decay(1, 1, 1) == 1.0

    # test neighborhood functions
    # zero distance should give 1.0 and long distances should approach 0.0
    @test SelfOrganizingMaps.gaussian_neighbor(0.0, 1.0) == 1.0
    @test SelfOrganizingMaps.gaussian_neighbor(10000.0, 1.0) ≈ 0.0

    @test SelfOrganizingMaps.mexicanhat_neighbor(0.0, 1.0) == 1.0
    @test SelfOrganizingMaps.mexicanhat_neighbor(10000, 1.0) ≈ 0.0

    k = 5
    nfeatures = 3
    som = SOM(k, nfeatures, 0.5, 0.25^2, :rectangular, :exponential, :exponential, :gaussian, euclidean, 1)
    @test size(som.W) == (nfeatures, k^2)
    @test size(som.coords) == (2, k^2)

    x = [1.0, 0.0, 0.0]
    @test typeof(SelfOrganizingMaps.getBMUidx(som, x)) <: Int

    # make sure update actually changes the weights
    W_prev = copy(som.W)
    SelfOrganizingMaps.updateWeights!(som, x, 1, 1)
    @test !all(W_prev .== som.W)

    # let's do the same with train
    W_prev = copy(som.W)
    X = [1.0 0.0 0.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0;
         ]
    train!(som, X)
    @test !all(W_prev .== som.W)
end


@testset "SelfOrganizingMaps.jl" begin
    X, y = make_regression(50, 3; rng=stable_rng());

    model = SelfOrganizingMap()
    m = machine(model, X)
    res = fit!(m)
    X̃ = MLJBase.transform(m, X)

    println(size(X̃))

    @test size(X̃) == (50, 2)

    fp = fitted_params(m)
    @test Set([:weights, :coords]) == Set(keys(fp))
    @test size(fp.coords) == (model.k^2, 2)
    @test size(fp.weights) == (model.k^2, 3)

    rpt = report(m)
    @test Set([:classes]) == Set(keys(rpt))
    @test size(rpt.classes, 1) == 50

    # predicted class is BMU (same as in report)
    classes = MLJBase.predict(m, X)
    @test scitype(classes[end]) <: Multiclass
    @test size(classes, 1) == 50
end


@testset "MLJ interface" begin
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    X = (; r, g, b) # a column table
    models = [SelfOrganizingMap,]
    failures, summary = MLJTestInterface.test(
        models,
        X;
        mod=@__MODULE__,
        verbosity=0, # set to 2 for maximum information
        throw=false, # set to true to debug
    )
    println(failures)
    println(summary[1].operations)
    @test isempty(failures)
    @test summary[1].operations == "predict, transform"
end


