using SelfOrganizingMaps
using Plots
using DataFrames
using ImageView #, Colors

# generating points on a sphere
# r,θ = getdiscpoints(1000)
# x = getpolar_x.(r, θ)
# y = getpolar_y.(r, θ)
# scatter(x, y, color=:blue, ms=1, aspect_ratio=:equal)

# ϕ, θ = getspherepoints(1000)
# x = getspherical_x.(ϕ,θ)
# y = getspherical_y.(ϕ,θ)
# z = getspherical_z.(ϕ,θ)
# scatter(x, y, z, ms=1, color=:blue, xlims=(-1,1), ylims=(-1,1))

# ϕ₁ = π/4
# ϕ₂ = π/4
# θ₁ = 0
# θ₂ = π/2

# sphericaldistance(ϕ₁,θ₁,ϕ₂,θ₂)*180/π
# sphericaldistance.(ϕ₁,θ₁,ϕ,θ)

# sphericaldistance(ϕ₁,θ₁,ϕ₂,θ₂)

# w = rand(2, 100)
# x = rand(2)
# i = getBMUidx(w, x)

# w[:,i]

# w = rand(2, 100, 100)
# i, j = getBMUidx(w, x)
# w[:,i,j]

# som = SquareSOM(2, 10,10)
# idx = getBMUidx(som, x)
# som.w[:,idx...]

# updateWeights!(som, x, som.η₀, som.σ₀²)

# practical example on random colors

M = 1
N =10
k = 3

som = SquareSOM(k, M, N; η₀=0.99, σ₀²=9.0, λ=0.1, β=0.1, nepochs=50)

function SOMtoRGB(som)
    SOM_pretty = [RGB(som.w[1,i,j], som.w[2,i,j], som.w[3,i,j]) for i∈1:M, j∈1:N]
    return SOM_pretty
end

function SOMtoRGB(som::SphericalSOM)
    SOM_pretty = [RGB(som.w[1,i], som.w[2,i], som.w[3,i]) for i∈1:size(som.w, 2)]
end



im1 = SOMtoRGB(som) # save this for before and after
imshow(im1)

# use random colors to train
X = DataFrame(rand(1000, 3), :auto)
train!(som, X)

im2 = SOMtoRGB(som)
imshow(im2)



Nfeatures = 10
NNodes = 1000

som = SphericalSOM(Nfeatures, NNodes)


x = getspherical_x.(som.ϕ, som.θ)
y = getspherical_y.(som.ϕ, som.θ)
z = getspherical_z.(som.ϕ, som.θ)

scatter(x,y,z)


updateWeights!(som, rand(10), 0.99, 4.0)


Nfeatures = 3
Nnodes = 500
X = DataFrame(rand(500, 3), :auto)
som = SphericalSOM(Nfeatures, Nnodes)
train!(som, X)


x = getspherical_x.(som.ϕ, som.θ)
y = getspherical_y.(som.ϕ, som.θ)
z = getspherical_z.(som.ϕ, som.θ)
size(x)

cs = SOMtoRGB(som)



plotlyjs()
scatter(x,y,z, color=cs)


