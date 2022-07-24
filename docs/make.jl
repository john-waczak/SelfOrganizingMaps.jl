using SelfOrganizingMaps
using Documenter

DocMeta.setdocmeta!(SelfOrganizingMaps, :DocTestSetup, :(using SelfOrganizingMaps); recursive=true)

makedocs(;
    modules=[SelfOrganizingMaps],
    authors="John Waczak",
    repo="https://github.com/john-waczak/SelfOrganizingMaps.jl/blob/{commit}{path}#{line}",
    sitename="SelfOrganizingMaps.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/SelfOrganizingMaps.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/SelfOrganizingMaps.jl",
    devbranch="main",
)
