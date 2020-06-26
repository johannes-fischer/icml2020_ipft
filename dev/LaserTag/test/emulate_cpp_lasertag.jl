# try to emulate the output of $ ./lasertag -r 7
using LaserTag
using POMDPs
using Test

let
    model = LaserTagPOMDP{DESPOTEmu, DMeas}(
        obstacles = Set{Coord}(
            Coord(xy...) for xy in [
                (3,1),
                (6,4),
                (10,4),
                (10,5),
                (2,6),
                (6,6),
                (1,7),
                (5,7)
            ]
        )                     
    )

    is = LTState(Coord(1,1), Coord(10,7), false)

    show(stdout, MIME("text/plain"), LaserTagVis(model, s=is))

    a = Dict{Symbol, Int}(:North=>1, :South=>2, :East=>3, :West=>4, :Tag=>5)

    aspor = [
        (a[:South], LTState([1,2], [11,7], false), DMeas(4, 9, 0, 0, 6, 2, 0, 0), -1.0),
        (a[:East], LTState([2,2], [11,7], false), DMeas(1, 8, 0, 0, 2, 0, 1, 1), -1.0),
        (a[:East], LTState([3,2], [11,7], false), DMeas(5, 8, 0, 0, 7, 1, 1, 0), -1.0),
        (a[:East], LTState([4,2], [11,7], false), DMeas(4, 7, 0, 0, 0, 1, 0, 3), -1.0),
        (a[:East], LTState([5,2], [11,7], false), DMeas(2, 3, 0, 2, 4, 0, 0, 5), -1.0),
        (a[:East], LTState([6,2], [11,7], false), DMeas(0, 3, 1, 3, 2, 0, 2, 5), -1.0),
        (a[:East], LTState([7,2], [11,7], false), DMeas(4, 4, 0, 6, 4, 2, 0, 6), -1.0),
        (a[:South], LTState([7,3], [11,7], false), DMeas(4, 2, 0, 6, 4, 4, 2, 0), -1.0),
        (a[:South], LTState([7,4], [11,7], false), DMeas(2, 2, 3, 0, 3, 4, 3, 4), -1.0),
        (a[:South], LTState([7,5], [11,7], false), DMeas(1, 2, 4, 4, 2, 6, 0, 0), -1.0),
        (a[:East], LTState([8,5], [11,7], false), DMeas(1, 1, 2, 5, 2, 4, 6, 1), -1.0),
        (a[:South], LTState([8,6], [11,7], false), DMeas(1, 2, 4, 0, 0, 0, 1, 2), -1.0),
        (a[:East], LTState([9,6], [11,7], false), DMeas(0, 2, 4, 0, 1, 0, 6, 2), -1.0),
        (a[:East], LTState([10,6], [11,7], false), DMeas(0, 0, 0, 1, 0, 1, 4, 0), -1.0),
        (a[:South], LTState([10,7], [11,7], false), DMeas(0, 0, 1, 2, 1, 0, 9, 0), -1.0),
        (a[:East], LTState([11,7], [11,7], false), LaserTag.D_SAME_LOC, -1.0),
        (a[:Tag], LTState([11,7], [11,7], true), LaserTag.D_SAME_LOC, 10.0),
    ]

    oprobs = [
        0.00258604,
        0.000232729,
        0.000177984,
        0.000319759,
        1.58596e-05,
        1.2223e-05,
        3.82869e-05,
        3.78645e-05,
        0.000100641,
        0.000288588,
        1.19013e-05,
        0.00023327,
        0.000348013,
        0.000925637,
        0.000897613,
        1,
        1
    ]

    discounted_rsum = 0.0
    disc = 1.0
    @show discount(model)
    for i in 1:length(aspor)
        a, sp, o, r = aspor[i]
        if i == 1
            s = is
        else
            s = aspor[i-1][2]
        end
        od = observation(model, s, a, sp)
        @test isapprox(pdf(od, o), oprobs[i], rtol=0.001)
        @test isapprox(reward(model, s, a, sp), r, atol=0.0001)
        @test !isterminal(model, s)
        discounted_rsum = discounted_rsum + disc*r
        @show discounted_rsum
        disc*=discount(model)
    end
    @show discounted_rsum

    @test isterminal(model, aspor[end][2])
    @test isapprox(discounted_rsum, -6.7962, atol=0.0001)

    println("Matches CPP!")
end
