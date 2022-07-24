using Plots
using DataFrames
using Random
using ImageView, Colors


# set up epochs
epochs = 0:50

# set up η(t) decay rate
λs = [0.001, 0.1, 0.5, 0.99]

p = plot();
xlabel!(p, "epoch")
ylabel!(p, "η(t)")
for λ ∈ λs
    η = exp.(-λ .* epochs)
    plot!(p, epochs, η, label="λ=$(λ)")
end
display(p)




# neighborhood distance function
distance = 0:30  # distance between nodes
σ² = [0.1, 1, 10, 100]  # radius

p2 = plot();
xlabel!(p2, "Distance")
ylabel!(p2, "Neighborhood function f")
for s² ∈ σ²
    f = exp.(-distance .^ 2 ./ (2*s²))
    plot!(p2, distance, f, label="σ²=$(s²)")
end
display(p2)



# -------------------------------------------------------

# set up test SOM
M = 100
N = 100
k = 3
SOM = rand(M, N, k)
x = rand(k)
D² = [sum((SOM[i,j,:] - x) .^ 2) for i∈1:M, j∈1:N]



# create a function to return the index (g, h) of the BMU in the SOM grid
function getBMU(SOM, x)
    M = size(SOM)[1]
    N = size(SOM)[2]

    # NOTE: it is very inefficient to have to allocate this array each time
    D² = [sum((SOM[i,j,:] - x) .^ 2) for i∈1:M, j∈1:N]
    gh = argmin(D²)
    return gh[1], gh[2]
end

# verify that it works
g,h = getBMU(SOM, x)



# create a function to update the weights
function updateWeights!(SOM, x, η, σ², bmu)
    M = size(SOM)[1]
    N = size(SOM)[2]

    g,h = bmu

    # if radius is small, only update bmu
    # otherwise change in cells via neighborhood function
    if σ² < 1e-3
        SOM[g,h,:] += η .* (x .- SOM[g, h, :])
    else
        for i ∈ 1:M, j ∈ 1:N
            d² = (i-g)^2 + (j-h)^2
            f = exp(-d²/(2σ²))
            SOM[i,j,:] += η .* f .* (x .- SOM[i,j,:])
        end
    end
end


# verify that it works
SOM1 = copy(SOM)

updateWeights!(SOM, x, 0.5, 10, (g,h))

SOM ≈ SOM1  # <--- this should be false



X = DataFrame(rand(500, 3), :auto)
data = Matrix(X)

# Create a function for the  training
function train!(SOM, df; η₀ = 0.1, σ₀² = 1, λ=0.1, β= 0.1, epochs=10)
    X = Matrix(df)
    η=η₀
    σ²=σ₀²

    # training loop
    for epoch ∈ epochs
        shuffle!(X)
        # loop through each datapoint
        for x ∈ eachrow(X)
            g, h = getBMU(SOM, x)
            updateWeights!(SOM, x, η, σ², (g,h))
        end

        # update the η and σ²
        η = η₀ * exp(-epoch * λ)
        σ² = σ₀² * exp(-epoch * β)
    end
end

# let's try it out
SOM1 = copy(SOM)
train!(SOM, X)

SOM ≈ SOM1  # looks like it worked to me!



# practical example on random colors

# set up SOM grid
M =50
N =50
k = 3
SOM = rand(M, N, k)

function SOMtoRGB(SOM)
    SOM_pretty = [RGB(SOM[i,j,1], SOM[i,j,2], SOM[i,j,3]) for i∈1:M, j∈1:N]
    return SOM_pretty
end

#imshow(SOMtoRGB(SOM))


im1 = SOMtoRGB(SOM) # save this for before and after

# use random colors to train
X = DataFrame(rand(5000, 3), :auto)
train!(SOM, X; η₀=0.99, σ₀²=10, λ=0.1, β=0.1,  epochs=250)

im2 = SOMtoRGB(SOM)

imshow(im1)
imshow(im2)
