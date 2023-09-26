using ACEluxpots
using Documenter

DocMeta.setdocmeta!(ACEluxpots, :DocTestSetup, :(using ACEluxpots); recursive=true)

makedocs(;
    modules=[ACEluxpots],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ortner/ACEluxpots.jl/blob/{commit}{path}#{line}",
    sitename="ACEluxpots.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ortner.github.io/ACEluxpots.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ortner/ACEluxpots.jl",
    devbranch="main",
)
