# Orthogonal projection onto {v | Av = 0, vᵢ = 0 for i ∈ fixed}, computed densely as a
# reference for the SubspaceProjector.
function reference_projection(A, fixed, x)
    n = size(A, 2)
    B = isempty(fixed) ? A : vcat(A, Matrix{Float64}(I, n, n)[fixed, :])
    return x - pinv(B) * (B * x)     # pinv: B may be rank deficient (redundant bounds)
end

# Schur complement H = I - GᵀG, with G = L₁₁⁻¹A_𝒜, whose Cholesky factor is maintained in
# the SubspaceProjector
function reference_schur(A, fixidx)
    G = cholesky(A * A').L \ A[:, fixidx]
    return I - G' * G
end

factor_LH(P) = LowerTriangular(P.LH[1:P.p, 1:P.p])

# To measure the allocations of `mul!`
projection_allocs(r, P, x) = @allocated mul!(r, P, x)

@testset "Subspace projector: construction and projection" begin
    m, n = 4, 8
    A = rand(m, n)
    chol_aat = cholesky(A * A')
    x = collect(1.0:n)

    P = Traulls.SubspaceProjector(A, chol_aat)

    @test P.W ≈ chol_aat.L \ A
    @test P.p == 0 && isempty(P.fixidx) && all(.!P.fixvars)
    @test Traulls.nb_degrees_of_freedom(P) == n - m

    proj_x = P * x
    @test proj_x ≈ reference_projection(A, Int[], x)
    @test norm(A * proj_x) < 1e-10
    @test P * proj_x ≈ proj_x

    # The output may alias the input
    y = copy(x)
    mul!(y, P, y)
    @test y ≈ proj_x

    @test_throws ErrorException Traulls.SubspaceProjector(rand(3, 3), cholesky(Matrix(1.0I, 3, 3)))
end

@testset "Subspace projector: activating bounds" begin
    m, n = 4, 8
    A = rand(m, n)
    x = collect(1.0:n)

    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    active = [3, 1, 6]                          # deliberately unsorted
    Traulls.set_active!(P, active)

    @test all(P.fixvars[active]) && all(.!P.fixvars[setdiff(1:n, active)])
    @test findall(i -> Traulls.is_fixed(P, i), 1:n) == sort(active)
    @test Traulls.nb_degrees_of_freedom(P) == n - m - length(active)

    # Factor rows follow the activation order
    @test P.p == length(active) && P.fixidx == active
    @test all(P.fixpos[active] .== 1:length(active))
    @test factor_LH(P) * factor_LH(P)' ≈ reference_schur(A, active)

    proj_x = P * x
    @test proj_x ≈ reference_projection(A, active, x)
    @test norm(proj_x[active]) < 1e-12
    @test norm(A * proj_x) < 1e-10
    @test P * proj_x ≈ proj_x

    # Activating an already active bound changes nothing
    Traulls.set_active!(P, [1])
    @test P.p == length(active) && P.fixidx == active

    # Constructor with an initial active set
    fixvars = falses(n); fixvars[active] .= true
    P2 = Traulls.SubspaceProjector(A, fixvars, cholesky(A * A'))
    @test P2.fixvars == fixvars && P2.p == length(active)
    @test P2 * x ≈ proj_x
end

@testset "Subspace projector: freeing bounds" begin
    m, n = 3, 10
    A = rand(m, n)
    x = randn(n)

    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    Traulls.set_active!(P, [7, 2, 5, 9, 4])

    # Remove a middle row of the factor
    Traulls.set_free!(P, [5])
    @test !Traulls.is_fixed(P, 5) && P.fixpos[5] == 0
    @test P.fixidx == [7, 2, 9, 4]
    @test all(P.fixpos[P.fixidx] .== 1:P.p)
    @test factor_LH(P) * factor_LH(P)' ≈ reference_schur(A, P.fixidx)
    @test P * x ≈ reference_projection(A, P.fixidx, x)

    # Remove the first and the last rows
    Traulls.set_free!(P, [7, 4])
    @test P.fixidx == [2, 9]
    @test factor_LH(P) * factor_LH(P)' ≈ reference_schur(A, P.fixidx)
    @test P * x ≈ reference_projection(A, [2, 9], x)

    # Freeing a free variable changes nothing
    Traulls.set_free!(P, [1])
    @test P.fixidx == [2, 9]

    # Freeing all the bounds restores the projector onto null(A)
    Traulls.set_free!(P, [2, 9])
    @test P.p == 0 && all(.!P.fixvars)
    @test P * x ≈ reference_projection(A, Int[], x)
end

@testset "Subspace projector: random updates and capacity growth" begin
    m, n = 5, 60
    A = randn(m, n)
    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    initial_capacity = size(P.LH, 1)

    for _ in 1:200
        if rand() < 0.65 && Traulls.nb_degrees_of_freedom(P) > 0
            Traulls.set_active!(P, rand(1:n))
        else
            Traulls.set_free!(P, rand(1:n))
        end
    end
    Traulls.set_active!(P, 1:40)                # forces the factor beyond its capacity

    fixed = findall(P.fixvars)
    @test size(P.LH, 1) > initial_capacity
    @test P.p == length(fixed) == length(P.fixidx)
    @test sort(P.fixidx) == fixed
    @test factor_LH(P) * factor_LH(P)' ≈ reference_schur(A, P.fixidx)
    x = randn(n)
    @test P * x ≈ reference_projection(A, fixed, x)
end

@testset "Subspace projector: redundant bounds" begin
    # The first row x₁ + x₂ = 0 makes the bound on x₂ redundant once x₁ is fixed
    n = 6
    A = vcat([1.0 1.0 0 0 0 0], rand(1, n))
    x = collect(1.0:n)

    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    Traulls.set_active!(P, [1, 2])

    @test Traulls.is_fixed(P, 2)                # flagged as active
    @test P.p == 1 && P.fixidx == [1]           # left out of the factor
    @test Traulls.nb_degrees_of_freedom(P) == n - 2 - 1
    @test P * x ≈ reference_projection(A, [1, 2], x)
    @test abs((P * x)[2]) < 1e-12

    # Freeing x₁ makes the bound on x₂ independent: it must enter the factor
    Traulls.set_free!(P, [1])
    @test Traulls.is_fixed(P, 2) && P.fixidx == [2]
    @test Traulls.nb_degrees_of_freedom(P) == n - 2 - 1
    @test P * x ≈ reference_projection(A, [2], x)
    @test abs((P * x)[2]) < 1e-12

    # Saturated subspace: further bounds are flagged but not factored
    m2, n2 = 2, 5
    A2 = rand(m2, n2)
    P2 = Traulls.SubspaceProjector(A2, cholesky(A2 * A2'))
    Traulls.set_active!(P2, 1:n2)
    @test Traulls.saturated_subspace(P2)
    @test P2.p == n2 - m2 && all(P2.fixvars)
    @test norm(P2 * randn(n2)) < 1e-10
end

@testset "Subspace projector: active set identification and reset" begin
    m, n = 3, 8
    A = rand(m, n)
    P = Traulls.SubspaceProjector(A, cholesky(A * A'))

    xlow = [0.0, -Inf, 0.0, 0.0, -1.0, -Inf, 0.0, 0.0]
    xupp = [1.0, Inf, Inf, 2.0, 1.0, Inf, Inf, Inf]
    x = [0.0, 3.0, 0.5, 2.0, 0.0, -4.0, 0.0, 1.0]    # active: 1, 4, 7

    Traulls.set_active!(P, [2, 3])
    Traulls.identify_active_set!(x, xlow, xupp, P)
    @test findall(P.fixvars) == [1, 4, 7]
    @test P.fixidx == [1, 4, 7]
    @test count(!iszero, P.fixpos) == 3
    v = randn(n)
    @test P * v ≈ reference_projection(A, [1, 4, 7], v)

    Traulls.reset_projector!(P)
    @test P.p == 0 && isempty(P.fixidx) && all(.!P.fixvars) && all(iszero, P.fixpos)
    @test P * v ≈ reference_projection(A, Int[], v)
end

@testset "Subspace projector: allocation-free projection" begin
    m, n = 4, 12
    A = rand(m, n)
    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    x, r = randn(n), zeros(n)

    projection_allocs(r, P, x)                  # warm up
    @test projection_allocs(r, P, x) == 0
    Traulls.set_active!(P, [2, 7, 11])
    projection_allocs(r, P, x)
    @test projection_allocs(r, P, x) == 0
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

    # In-place projection: returns its output, may alias the input, does not allocate
    w = collect(1.0:n)
    @test mul!(r, P, w) === r
    mul!(w, P, w)
    @test w ≈ r
    projection_allocs(r, P, v)
    @test projection_allocs(r, P, v) == 0

    # Reset subspace
    Traulls.reset_projector!(P)
    @test all(.!P.fixvars)
end

# To measure the allocations of `update_inner_active_set!`
inner_update_allocs(s, slow, supp, P) =
    @allocated Traulls.update_inner_active_set!(s, slow, supp, P)

@testset "Inner active set update" begin
    m, n = 3, 8
    A = rand(m, n)
    slow, supp = fill(-1.0, n), fill(1.0, n)
    # Bounds reached: 1 and 5 (lower, 5 within the tolerance), 3 and 8 (upper)
    s = [-1.0, 0.2, 1.0, 0.5, -1.0 + 1e-12, 0.0, 0.99, 1.0]
    active = [1, 3, 5, 8]

    for P in (Traulls.CoordinateSubspaceProjector(n),
              Traulls.SubspaceProjector(A, cholesky(A * A')))

        # Already fixed variables are kept, newly active ones are added
        Traulls.set_active!(P, [2])
        Traulls.update_inner_active_set!(s, slow, supp, P)
        @test findall(i -> Traulls.is_fixed(P, i), 1:n) == sort([2; active])

        # No allocation once the buffers of the projector have been used
        Traulls.reset_projector!(P)
        inner_update_allocs(s, slow, supp, P)
        Traulls.reset_projector!(P)
        @test inner_update_allocs(s, slow, supp, P) == 0
        @test findall(i -> Traulls.is_fixed(P, i), 1:n) == active
    end
end

# To measure the allocations of `mul!`
projection_allocs(r, P, x, alpha, beta) = @allocated mul!(r, P, x, alpha, beta)

@testset "Five-argument mul!" begin
    m, n = 3, 9
    A = rand(m, n)
    fixed = [2, 5, 8]

    PS = Traulls.SubspaceProjector(A, cholesky(A * A'))
    PC = Traulls.CoordinateSubspaceProjector(n)
    Traulls.set_active!(PS, fixed)
    Traulls.set_active!(PC, fixed)

    for P in (PS, PC)
        v, r0 = randn(n), randn(n)
        alpha, beta = -1.7, 0.3
        Pv = P * v

        # r ← αPv + βr
        r = copy(r0)
        @test mul!(r, P, v, alpha, beta) === r
        @test r ≈ alpha * Pv + beta * r0

        # β = 0: the input content of r is not read, even if it is NaN
        r = fill(NaN, n)
        mul!(r, P, v, alpha, 0.0)
        @test r ≈ alpha * Pv

        # α = 0 only scales r
        r = copy(r0)
        mul!(r, P, v, 0.0, beta)
        @test r ≈ beta * r0

        # r may alias v, also when β ≠ 0
        w = copy(v)
        mul!(w, P, w, alpha, beta)
        @test w ≈ alpha * Pv + beta * v

        # The 3-argument method is the case α = 1, β = 0
        @test mul!(similar(v), P, v) ≈ mul!(similar(v), P, v, 1.0, 0.0) ≈ Pv

        # No allocation, including with a Bool or Int scalar
        r = copy(r0)
        projection_allocs(r, P, v, alpha, beta)
        @test projection_allocs(r, P, v, alpha, beta) == 0
        projection_allocs(r, P, v, true, false)
        @test projection_allocs(r, P, v, true, false) == 0
        @test mul!(similar(v), P, v, 2, 0) ≈ 2 * Pv
    end
end
