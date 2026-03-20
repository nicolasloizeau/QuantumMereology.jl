using QuantumMereology
using Documenter

DocMeta.setdocmeta!(QuantumMereology, :DocTestSetup, :(using QuantumMereology); recursive=true)

makedocs(;
    modules=[QuantumMereology],
    authors="Nicolas Loizeau",
    sitename="QuantumMereology.jl",
    format=Documenter.HTML(;
        canonical="https://nicolasloizeau.github.io/QuantumMereology.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nicolasloizeau/QuantumMereology.jl",
    devbranch="main",
)
