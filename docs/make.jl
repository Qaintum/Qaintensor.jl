using Documenter, Qaintessent, Qaintensor

makedocs(
    sitename="Qaintensor.jl Documentation",
    pages = [
        "Home" => "index.md",
        "Section" => [
            "tensors.md",
            "mps.md",
            "mpo.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/Qaintum/Qaintensor.jl.git",
    push_preview = true
)
