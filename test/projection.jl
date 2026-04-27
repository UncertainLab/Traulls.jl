@testset "Subspace matrix" begin

    m=4; n = 8
    A = rand(m,n)
    chol_aat = cholesky(A*A')
    B = Traulls.SubspaceMatrix(A)

    @test B.eqmat ≈ A
    @test all(.!(B.fixvars))
    fix_bounds = [1,3,5,7]
    Traulls.add_subspace!(B, fix_bounds)
    @test all(B.fixvars[fix_bounds]) && all(.!(B.fixvars[setdiff(1:n,fix_bounds)]))
    p = size(fix_bounds,1)

    
    Z = Matrix{Float64}(I,n,n)[fix_bounds,:]
    greedy_B = vcat(A,Z)
    
    x = Vector{Float64}(collect(1:n))
    xt = Vector{Float64}(collect(1:m+p))
    
    res = B*x
    res_tr = Traulls.transpose(B)*xt
    
    @test size(res,1) == m+p
    @test size(res_tr,1) == n
    @test res ≈ greedy_B*x
    @test res_tr ≈ greedy_B'xt

    # try with all bounds inactive
    B.fixvars .= false
    xt = Vector{Float64}(collect(1:m))
    res = B*x
    res_tr = Traulls.transpose(B)*xt

    @test Traulls.nb_fixed(B) == 0
    @test size(res, 1) == m
    @test size(res_tr, 1) == n
    @test res ≈ A*x
    @test res_tr ≈ A'xt
end

@testset "Subspace projections" begin
    m=4; n = 8
    A = rand(m,n)
    chol_aat = cholesky(A*A')
    active = [1, 3]
    fix_bounds = BitVector([true,false,true,false,false,false,false,false])
    
    Z = Matrix{Float64}(I,n,n)[findall(fix_bounds),:]
    greedy_B = vcat(A,Z)

    scratch_chol = Traulls.cholesky_augmented_gram_mat(A, fix_bounds, chol_aat)

    P = Traulls.SubspaceProjector(A, chol_aat)

    @test all(.!(P.workspace_mat.fixvars))
    Traulls.set_active!(P, active)

    @test all(P.workspace_mat.fixvars[active]) &&
        all(.!(P.workspace_mat.fixvars[setdiff(1:n, active)]))
    @test findall(i -> Traulls.is_fixed(P, i), 1:n) == active
    @test Traulls.nb_degrees_of_freedom(P) == n - m - size(active, 1)

    x = Vector{Float64}(collect(1:n))
    proj_x = Vector{Float64}(undef,n)

    Traulls.mul!(proj_x, P, x)

    @test P.chol_gram_augmat.L ≈ scratch_chol.L
    @test P.workspace_mat*x ≈ greedy_B*x
    @test norm(proj_x[findall(fix_bounds)]) < 1e-12
    @test norm(P.workspace_mat*proj_x) < 1e-12

    Traulls.set_free!(P, active)
    @test P.workspace_mat.eqmat ≈ A
    @test all(i -> !Traulls.is_fixed(P, i), 1:n)
end

@testset "Coordinate subspace projector" begin

    n = 10

    # Initialize
    P = Traulls.CoordinateSubspaceProjector(n)
    v = ones(n)

    @test all(.!P.fixvars)
    @test P*v ≈ v

    # Set some bounds active
    fixed = collect(1:2:n)
    Traulls.set_active!(P, fixed)

    @test all(P.fixvars[fixed]) && all(.!P.fixvars[setdiff(1:n,fixed)])
    @test Traulls.nb_degrees_of_freedom(P) == n - size(fixed,1)
    @test findall(i -> Traulls.is_fixed(P, i), 1:n) == fixed


    r = P*v

    @test all(isapprox(0.0), r[P.fixvars]) && r[.!P.fixvars] ≈ v[.!P.fixvars]

    # Reset subspace
    Traulls.reset_projector!(P)
    @test all(.!P.fixvars)


end
