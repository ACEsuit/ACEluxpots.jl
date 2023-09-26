using ACEluxpots
using Documenter

DocMeta.setdocmeta!(ACEluxpots, :DocTestSetup, :(using ACEluxpots); recursive=true)

makedocs(;
    modules=[ACEluxpots],
    authors="Christoph Ortner <christophortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEluxpots.jl/blob/{commit}{path}#{line}",
    sitename="ACEluxpots.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEluxpots.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACEluxpots.jl",
    devbranch="main",
)
