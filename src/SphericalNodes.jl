# see: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

"""
    getdiscpoints(N::Int)

Generate a set of `N` (nearly) equally distributed points within a circle of radius 1.
"""
function getdiscpoints(N::Int)
    idx = (0:1:(N-1)) .+ 0.5

    r = sqrt.(idx/N)
    φ = (1+√5)/2
    θ = 2π .* φ .* idx

    return r, θ
end


"""
    getspherepoints(N::Int)

Generate `N` (nearly) evenly distributed points on the unit sphere using the fibonacci spiral procedure.
"""
function getspherepoints(N::Int)
    idx = (0:1:(N-1)) .+ 0.5

    θ = acos.(1.0 .- 2.0 .* idx ./ N)

    φ = (1+√5)/2
    ϕ = 2π .* φ .* idx

    return ϕ, θ
end



getpolar_x(r,θ) = r*cos(θ)
getpolar_y(r,θ) = r*sin(θ)

getspherical_x(r,ϕ,θ) = r*cos(ϕ)*sin(θ)
getspherical_y(r,ϕ,θ) = r*sin(ϕ)*sin(θ)
getspherical_z(r,ϕ,θ) = r*cos(θ)

getspherical_x(ϕ,θ) = cos(ϕ)*sin(θ)
getspherical_y(ϕ,θ) = sin(ϕ)*sin(θ)
getspherical_z(ϕ,θ) = cos(θ)


