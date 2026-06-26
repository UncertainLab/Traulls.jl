# Orthogonal projection onto {v | Av = 0, vᵢ = 0 for i ∈ fixed}, computed
# directly (and order-independently) as a reference for the incremental
# SubspaceProjector.
function reference_projection(A, fixed, x)
    n = size(A, 2)
    Z = Matrix{Float64}(I, n, n)[fixed, :]
    B = isempty(fixed) ? A : vcat(A, Z)
    return x - B' * ((B * B') \ (B * x))
end

# Measured through a function barrier so the result reflects only mul!'s own
# allocations (used to check the in-place projection is allocation-free).
projection_allocs(r, P, x) = @allocated mul!(r, P, x)

@testset "Subspace projector — construction and projection" begin
    m, n = 4, 8
    A = rand(m, n)
    chol_aat = cholesky(A * A')
    x = collect(1.0:n)

    P = Traulls.SubspaceProjector(A, chol_aat)

    # No bound active: projection onto null(A)
    @test all(.!P.fixvars)
    @test Traulls.nb_degrees_of_freedom(P) == n - m
    proj_x = P * x
    @test proj_x ≈ reference_projection(A, Int[], x)
    @test norm(A * proj_x) < 1e-10            # lies in null(A)
    @test P * proj_x ≈ proj_x                 # idempotent
end

@testset "Subspace projector — active set management" begin
    m, n = 4, 8
    A = rand(m, n)
    chol_aat = cholesky(A * A')
    x = collect(1.0:n)

    P = Traulls.SubspaceProjector(A, chol_aat)
    active = [3, 1, 6]                         # deliberately unsorted
    Traulls.set_active!(P, active)

    @test all(P.fixvars[active]) && all(.!P.fixvars[setdiff(1:n, active)])
    @test findall(i -> Traulls.is_fixed(P, i), 1:n) == sort(active)
    @test Traulls.nb_degrees_of_freedom(P) == n - m - length(active)

    proj_x = P * x
    @test proj_x ≈ reference_projection(A, active, x)
    @test norm(proj_x[active]) < 1e-12         # fixed components vanish
    @test norm(A * proj_x) < 1e-10             # still in null(A)
    @test P * proj_x ≈ proj_x                  # idempotent

    # Freeing the bounds restores the null(A) projector
    Traulls.set_free!(P, active)
    @test all(.!P.fixvars)
    @test Traulls.nb_degrees_of_freedom(P) == n - m
    @test P * x ≈ reference_projection(A, Int[], x)
end

@testset "Subspace projector — order independence and reset" begin
    m, n = 3, 9
    A = rand(m, n)
    chol_aat = cholesky(A * A')
    x = randn(n)
    active = [7, 2, 5, 8]

    # Activating in different orders yields the same projection
    P1 = Traulls.SubspaceProjector(A, chol_aat)
    Traulls.set_active!(P1, active)

    P2 = Traulls.SubspaceProjector(A, chol_aat)
    for j in reverse(active)
        Traulls.set_active!(P2, [j])
    end

    P3 = Traulls.SubspaceProjector(A, BitVector([i in active for i in 1:n]), chol_aat)

    ref = reference_projection(A, active, x)
    @test P1 * x ≈ ref
    @test P2 * x ≈ ref
    @test P3 * x ≈ ref

    # reset_projector! returns to the all-free state and is reusable
    Traulls.reset_projector!(P1)
    @test all(.!P1.fixvars) && Traulls.nb_degrees_of_freedom(P1) == n - m
    @test P1 * x ≈ reference_projection(A, Int[], x)
    Traulls.set_active!(P1, [4])
    @test P1 * x ≈ reference_projection(A, [4], x)
end

@testset "Subspace projector — in-place mul! is allocation free" begin
    # Large n: the previous implementation allocated O(n) temporaries per call;
    # the rewritten one only uses preallocated buffers, so allocations are O(1).
    m, n = 20, 200
    A = rand(m, n)
    chol_aat = cholesky(A * A')
    P = Traulls.SubspaceProjector(A, chol_aat)
    Traulls.set_active!(P, collect(2:2:60))

    x = randn(n)
    r = similar(x)
    mul!(r, P, x)                              # warm up / compile
    @test projection_allocs(r, P, x) < 256     # independent of n
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
