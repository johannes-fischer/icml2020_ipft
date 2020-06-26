using Documenter
using POMDPSimulators

makedocs(
    format = :html,
    sitename = "POMDPSimulators.jl"
)

deploydocs(
    repo = "github.com/JuliaPOMDP/POMDPSimulators.jl.git",
    julia = "1.0",
    target = "build",
    deps = nothing,
    make = nothing
)

