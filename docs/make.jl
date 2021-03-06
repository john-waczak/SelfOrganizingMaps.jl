using Documenter
using SelfOrganizingMaps

makedocs(
    sitename = "SelfOrganizingMaps",
    format = Documenter.HTML(),
    modules = [SelfOrganizingMaps]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
