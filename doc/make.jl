using Documenter, Qaintensor

makedocs(
    sitename="Qaintensor.jl Documentation",
    pages = [
        "Home" => "index.md",
        "Section" => [
            "index.md",            
            "mps.md",
            "mpo.md",
	    "tensors.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/Qaintum/Qaintensor.jl.git",
    push_preview = true
)
