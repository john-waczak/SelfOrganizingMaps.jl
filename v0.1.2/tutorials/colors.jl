using SelfOrganizingMaps
using Plots
using MLJ


function SOMtoRGB(som::SOM)
    SOM_pretty = [RGB(som.W[1,i], som.W[2,i], som.W[3,i]) for i∈1:size(som.W, 2)]
end

function plotSquareSOM(som::SOM)
    cs = SOMtoRGB(som) # save this for before and after
    p = scatter(som.coords[1,:], som.coords[2,:];
                xlims=(-0.05, 1.05),
                ylims=(-0.05, 1.05),
                color=cs,
                marker=:rect,
                 ms=12.0,
                aspect_ratio=:equal,
                label=""
                )
    return p
end

function plotHexSOM(som::SOM)
    cs = SOMtoRGB(som) # save this for before and after
    p = scatter(som.coords[1,:], som.coords[2,:];
                #xlims=(-0.05, 1.05),
                #ylims=(-0.05, 0.85),
                color=cs,
                marker=:hexagon,
                #ms=6.0,
                ms=9.0,
                aspect_ratio=:equal,
                label="",
                ticks=nothing,
                xguide="",
                yguide="",
                framestyle=:none,
                dpi=600,
                )
    return p
end


r = [1.0, 0.0, 0.0]
g = [0.0, 1.0, 0.0]
b = [0.0, 0.0, 1.0]
X = (; r, g, b) # a column table

model = SelfOrganizingMap()
model.k = 25
model.η = 0.20
model.σ² = 0.2^2
model.grid_type= :hexagonal
model.Nepochs=100

m = machine(model, X)
fit!(m)

m.fitresult

size(fitted_params(m).weights)
size(fitted_params(m).coords)

report(m)

X̂ = transform(m, X)


plotHexSOM(m.fitresult)
savefig("SOM_demo.png")
savefig("SOM_demo.svg")
