using SelfOrganizingMaps
using Plots
using DataFrames
using Distances

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




#X = rand(3, 5000)
X = [1.0 0.0 0.0;
     0.0 1.0 0.0;
     0.0 0.0 1.0;
     ]


#som = SOM(25, 3, 1.0, 0.2^2, :hexagonal, :asymptotic, :gaussian, euclidean, 1)
#som = SOM(25, 3, 0.20, 0.2^2, :hexagonal, :exponential, :gaussian, cosine_dist, 100)
som = SOM(25, 3, 0.20, 0.2^2, :hexagonal, :exponential, :gaussian, cosine_dist, 1)

p1 = plotHexSOM(som)
savefig("imgs/000.png")

for i ∈ 1:100
    train!(som, X)
    p = plotHexSOM(som)
    savefig("imgs/"*lpad(i,3,"0")*".png")
end



p2 = plotHexSOM(som)
# layout = @layout [a{0.45w} b{0.45w}]
plot(p1, p2, dpi=600) #, layout=layout)#, plot_title="Pre-training vs Post-training")

savefig("test_som.svg")
savefig("test_som.png")
savefig("test_som.pdf")
