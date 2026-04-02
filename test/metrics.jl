@testset "Function evaluation counters" begin

    counters = Traulls.TraullsCounters()

    counters_fields = fieldnames(Traulls.TraullsCounters)

    @test all(f -> getfield(counters, f) == 0, counters_fields)

    counters.nres_eval += 1

    @test getfield(counters, :nres_eval) > 0

    Traulls.reset!(counters)

    @test all(f -> getfield(counters, f) == 0, counters_fields)
end
