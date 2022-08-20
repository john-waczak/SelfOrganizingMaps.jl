function getspherepoints(N::Int)
    idx = (0:1:(N-1)) .+ 0.5
    φ = (1+√5)/2  # golden ratio

    # using physics convention
    ϕ = π/2 .- acos.(1.0 .- 2.0 .* idx ./ N)  # latitude
    λ = 2π .* φ .* idx # polar angle (longitude)

    # return vcat(λ', ϕ')
    return vcat(ϕ', λ')
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

    X = [x for x ∈ x, y ∈ y]
    Y = [y for x ∈ x, y ∈ y]

    # offset every other row of the x's
    X[:, 1:2:end] .= X[:, 1:2:end] .+ Δ/2

    return vcat(vec(X)', vec(Y)')
end


getspherical_x(λ,ϕ) = cos(ϕ)*cos(λ)
getspherical_y(λ,ϕ) = cos(ϕ)*sin(λ)
getspherical_z(λ,ϕ) = sin(ϕ)

