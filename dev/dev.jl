using SelfOrganizingMaps
using Plots


r,θ = getdiscpoints(1000)

x = getpolar_x.(r, θ)
y = getpolar_y.(r, θ)
scatter(x, y, color=:blue, ms=1, aspect_ratio=:equal)



ϕ, θ = getspherepoints(1000)
XYZ =  sphericaltocartesian.(ϕ, θ)
x = [xyz[1] for xyz ∈ XYZ]
y = [xyz[2] for xyz ∈ XYZ]
z = [xyz[3] for xyz ∈ XYZ]


scatter(x, y, z, ms=1, color=:blue, xlims=(-1,1), ylims=(-1,1))
