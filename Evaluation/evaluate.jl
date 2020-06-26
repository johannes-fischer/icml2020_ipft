using DataFrames
using CSV
using Statistics

filenames = [

    ### Put result table files here for reviewing the results, e.g.

    # LD 10
    # "Experiments/results/ld10_2020-02-19_003851.csv"

    # LD 3
    # "Experiments/results/ld3_2020-02-19_050959.csv"

    # CLD 10
    # "Experiments/results/cld10_2020-02-19_015442.csv"

    # CLD 3
    # "Experiments/results/cld3_2020-02-19_030803.csv"

    ]

data = vcat([CSV.read(fname) for fname in filenames]...)

r = by(data, :solver, N = :reward => length, mean = :reward => mean, std = :reward => r->std(r)/sqrt(length(r)))
sort!(r, :mean, rev=true)
display(filenames)
@show r
