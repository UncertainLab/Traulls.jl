@testset "Subspace matrix" begin

    m=4; n = 8
    A = rand(m,n)
    chol_aat = cholesky(A*A')
    B = Traulls.SubspaceMatrix(A)

    @test B.eqmat ≈ A
    @test all(.!(B.fixvars))
    fix_bounds = [1,3,5,7]
    Traulls.update_subspace!(B,fix_bounds)
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
    
    @test size(res,1) == m
    @test size(res_tr,1) == n
    @test res ≈ A*x
    @test res_tr ≈ A'xt
end

@testset "Subspace projections" begin
    m=4; n = 8
    A = rand(m,n)
    chol_aat = cholesky(A*A')
    active = [1,3]
    fix_bounds = BitVector([true,false,true,false,false,false,false,false])
    
    Z = Matrix{Float64}(I,n,n)[findall(fix_bounds),:]
    greedy_B = vcat(A,Z)

    scratch_chol = Traulls.cholesky_augmented_gram_mat(A,fix_bounds,chol_aat)

    P = Traulls.SubspaceProjector(A,chol_aat)

    @test all(.!(P.workspace_mat.fixvars))
    Traulls.update_subspace_projector!(P,active)

    @test all(P.workspace_mat.fixvars[active]) &&
        all(.!(P.workspace_mat.fixvars[setdiff(1:n,active)]))

    x = Vector{Float64}(collect(1:n))
    proj_x = Vector{Float64}(undef,n)

    Traulls.mul!(proj_x,P,x)

    @test P.chol_gram_augmat.L ≈ scratch_chol.L
    @test P.workspace_mat*x ≈ greedy_B*x
    @test norm(proj_x[findall(fix_bounds)]) < 1e-12
    @test norm(P.workspace_mat*proj_x) < 1e-12
end
