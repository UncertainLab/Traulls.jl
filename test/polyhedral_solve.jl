# Integration tests exercising the SubspaceProjector through the full solver: problems
# with general linear equality constraints (nlincons > 0) select the SubspaceProjector.
@testset "Solve with linear equality constraints" begin

    n = 4
    nres = n
    ncons = 1

    # Residuals r(x) = x - target  ⇒  minimize ½‖x - target‖²
    A = reshape([1.0, 1.0, 1.0, 1.0], 1, n)   # x₁ + x₂ + x₃ + x₄ = 1
    b = [1.0]

    @testset "Inactive bounds" begin
        target = [1.0, 2.0, 3.0, 4.0]
        r!(rx, x) = (rx .= x .- target; nothing)
        jac_r!(J, x) = (J .= Matrix{Float64}(I, n, n); nothing)

        # Nonlinear equality constraint x₄ = 0
        c!(cx, x) = (cx[1] = x[4]; nothing)
        jac_c!(C, x) = (C .= [0.0 0.0 0.0 1.0]; nothing)

        xlow, xupp = fill(-10.0, n), fill(10.0, n)
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp, zeros(n),
                                   n, nres, ncons, Val(:only_equalities))
        @test model.nlincons == 1

        results = traulls(model; init_mult = false)
        sol = results.solution

        @test results.status == Traulls.first_order_critical
        @test results.feasibility ≤ 1e-5
        @test sol ≈ [-2/3, 1/3, 4/3, 0.0] atol = 1e-5
        @test A * sol ≈ b atol = 1e-6
    end

    @testset "Active bounds at the solution" begin
        # Projection of the target onto the simplex {x ≥ 0, Σxᵢ = 1} is (0, 1, 0, 0)
        target = [1.0, 2.0, -1.0, -3.0]
        r!(rx, x) = (rx .= x .- target; nothing)
        jac_r!(J, x) = (J .= Matrix{Float64}(I, n, n); nothing)

        # Nonlinear equality constraint x₃ + x₄² = 0, satisfied at the solution
        c!(cx, x) = (cx[1] = x[3] + x[4]^2; nothing)
        jac_c!(C, x) = (C .= [0.0 0.0 1.0 2x[4]]; nothing)

        xlow, xupp = zeros(n), ones(n)
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp, fill(0.25, n),
                                   n, nres, ncons, Val(:only_equalities))

        results = traulls(model)
        sol = results.solution

        @test results.status == Traulls.first_order_critical
        @test sol ≈ [0.0, 1.0, 0.0, 0.0] atol = 1e-5
        @test A * sol ≈ b atol = 1e-6
        @show sol
        @test all(xlow .- 1e-6 .<= sol .<= xupp .+ 1e-6)
    end
end

# Regression test: once x₂ ≤ 1 becomes active on the Cauchy path, the projected direction
# vanishes up to roundoff and must not produce a step leaving the feasible set.
@testset "Vanishing projected direction in the Cauchy step" begin

    n = 4
    nres = n
    ncons = 1

    # Minimize ½‖x - target‖² s.t. x₁ + x₂ + x₃ = 0, x₁ ≥ -1, x₂ ≤ 1, with x₄ free
    target = [-1.0, 4.0, -1.0, 1.0]
    A = [1.0 1.0 1.0 0.0]
    b = [0.0]
    r!(rx, x) = (rx .= x .- target; nothing)
    jac_r!(J, x) = (J .= Matrix{Float64}(I, n, n); nothing)

    # Nonlinear equality constraint x₄² - 1 = 0, satisfied at the solution
    c!(cx, x) = (cx[1] = x[4]^2 - 1.0; nothing)
    jac_c!(C, x) = (C .= [0.0 0.0 0.0 2x[4]]; nothing)

    xlow = [-1.0, -Inf, -Inf, -Inf]
    xupp = [Inf, 1.0, Inf, Inf]
    model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp,
                               [0.0, -2.0, 2.0, 1.0], n, nres, ncons, Val(:only_equalities))

    results = traulls(model)
    sol = results.solution

    @test results.status == Traulls.first_order_critical
    @test sol ≈ [-0.5, 1.0, -0.5, 1.0] atol = 1e-5
    @test A * sol ≈ b atol = 1e-10
    @test all(xlow .- 1e-10 .<= sol .<= xupp .+ 1e-10)
end
