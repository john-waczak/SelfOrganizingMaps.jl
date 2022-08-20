"""
    getspherepoints(N::Int)

Generate `N` (almost) evenly distributed points on the unit sphere using the fibonacci spiral procedure.
"""
function getspherepoints(N::Int)
    idx = (0:1:(N-1)) .+ 0.5
    φ = (1+√5)/2  # golden ratio

    # using physics convention
    ϕ = π/2 .- acos.(1.0 .- 2.0 .* idx ./ N)  # latitude
    λ = 2π .* φ .* idx # polar angle (longitude)

    return vcat(λ', ϕ')
end



function getsquarepoints(k)
    x = collect(range(0.0, stop=1.0, length=k))
    y = collect(range(0.0, stop=1.0, length=k))

    x̃ = vec([x for x ∈ x, y ∈ y])
    ỹ = vec([y for x ∈ x, y ∈ y])

    return vcat(x̃', ỹ')
end


# see: https://www.redblobgames.com/grids/hexagons/
function gethexpoints(k)
    Δ = 1/k
    x = collect(range(0.0, step=Δ, length=k))
    y = collect(range(0.0, step=Δ*sqrt(3)/2, length=k))
#    y = collect(range(0.0, step=Δ, length=k))

    X = [x for x ∈ x, y ∈ y]
    Y = [y for x ∈ x, y ∈ y]

    # offset every other row of the x's
    X[:, 1:2:end] .= X[:, 1:2:end] .+ Δ/2

    return vcat(vec(X)', vec(Y)')
end


getspherical_x(λ,ϕ) = cos(ϕ)*cos(λ)
getspherical_y(λ,ϕ) = cos(ϕ)*sin(λ)
getspherical_z(λ,ϕ) = sin(ϕ)


# # test spherical points
# X = getspherepoints(1000)
# λs = X[1, :]
# ϕs = X[2, :]
# x = getspherical_x.(λs, ϕs)
# y = getspherical_y.(λs, ϕs)
# z = getspherical_z.(λs, ϕs)

# scatter(x,y,z)


# # test square points
# X = getsquarepoints(10)
# scatter(X[1,:], X[2,:];
#         xlims=(-0.2, 1.2),
#         ylims=(-0.2, 1.2),
#         marker=:rect,
#         ms=12.0,
#         aspect_ratio=:equal)


# test hex points
# X = gethexpoints(10)
# scatter(X[1,:], X[2,:];
#         xlims=(-0.2, 1.2),
#         ylims=(-0.2, 1.2),
#         marker=:hexagon,
#         ms=14.0,
#         aspect_ratio=:equal)

