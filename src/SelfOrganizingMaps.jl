module SelfOrganizingMaps

using Plots

include("SphericalNodes.jl")
include("SOM.jl")

export getdiscpoints
export getspherepoints
export polartocartesian
export sphericaltocartesian

export getpolar_x
export getpolar_y

export getspherical_x
export getspherical_y
export getspherical_z

export sphericaldistance
export euclideandistance
export SOM
export SquareSOM
export SphericalSOM
export getBMUidx

end
