@testset "Workspace Allocations" begin

    n = 5
    m = 10
    p = 4


    workspace = Traulls.Workspace(Float64,n,m,p)

    @test eltype(workspace.x_prev) == Float64
    @test size(workspace.proj_g,1) == n
    @test size(workspace.rx_prev,1) == m
    @test size(workspace.cx_prev,1) == p

end
