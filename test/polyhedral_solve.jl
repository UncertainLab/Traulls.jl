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

        # Starting at the solution, a degenerate vertex where the four bounds and the
        # equality are active: the multipliers are not unique
        for eps_active in (0.0, 1e-3)
            model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp,
                                       [0.0, 1.0, 0.0, 0.0], n, nres, ncons,
                                       Val(:only_equalities))
            results = traulls(model; eps_active = eps_active)
            @test results.status == Traulls.first_order_critical
            @test results.solution ≈ [0.0, 1.0, 0.0, 0.0] atol = 1e-5
        end
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

# Regression tests: the vertex V = (-1, 1, 0) of {x₁ + x₂ + x₃ = 0, x₁ ≥ -1, x₂ ≤ 1} has a
# zero reduced gradient for any objective, but it is not critical when the multiplier of
# x₁ ≥ -1 is negative. The solver must neither stop there nor keep the bound fixed.
@testset "Release of bounds with multipliers of the wrong sign" begin

    n = 4
    nres = n
    ncons = 1

    A = [1.0 1.0 1.0 0.0]
    b = [0.0]
    xlow = [-1.0, -Inf, -Inf, -Inf]
    xupp = [Inf, 1.0, Inf, Inf]

    # Minimize ½‖x - target‖² over the polyhedron, with x₄ free and x₄² - 1 = 0
    function vertex_model(target, x0)
        r!(rx, x) = (rx .= x .- target; nothing)
        jac_r!(J, x) = (J .= Matrix{Float64}(I, n, n); nothing)
        c!(cx, x) = (cx[1] = x[4]^2 - 1.0; nothing)
        jac_c!(C, x) = (C .= [0.0 0.0 0.0 2x[4]]; nothing)
        Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp, x0, n, nres, ncons,
                           Val(:only_equalities))
    end

    @testset "Starting at the vertex, bounds inactive at the solution" begin
        # Multipliers at V: -5 for x₁ ≥ -1 (wrong sign), -1 for x₂ ≤ 1
        model = vertex_model([2.0, 0.0, -2.0, 1.0], [-1.0, 1.0, 0.0, 1.0])
        results = traulls(model)

        @test results.status == Traulls.first_order_critical
        @test results.solution ≈ [2.0, 0.0, -2.0, 1.0] atol = 1e-5
    end

    @testset "Reaching the vertex along the face x₁ = -1" begin
        # Multipliers at V: -1 for x₁ ≥ -1 (wrong sign), -4 for x₂ ≤ 1
        model = vertex_model([-1.0, 4.0, -1.0, 1.0], [-1.0, -2.0, 3.0, 1.0])
        results = traulls(model)
        sol = results.solution

        @test results.status == Traulls.first_order_critical
        @test sol ≈ [-0.5, 1.0, -0.5, 1.0] atol = 1e-5
        @test A * sol ≈ b atol = 1e-10
        @test all(xlow .- 1e-10 .<= sol .<= xupp .+ 1e-10)
    end

    @testset "ε-active set strategy" begin
        # Starting at the vertex, close to the vertex and along the face x₁ = -1
        for (target, x0, xsol) in (
                ([2.0, 0.0, -2.0, 1.0], [-1.0, 1.0, 0.0, 1.0], [2.0, 0.0, -2.0, 1.0]),
                ([-1.0, 4.0, -1.0, 1.0], [-1.0 + 1e-6, 1.0 - 3e-6, 2e-6, 1.0],
                 [-0.5, 1.0, -0.5, 1.0]),
                ([-1.0, 4.0, -1.0, 1.0], [-1.0, -2.0, 3.0, 1.0], [-0.5, 1.0, -0.5, 1.0]))
            results = traulls(vertex_model(target, x0); eps_active = 1e-3)
            sol = results.solution

            @test results.status == Traulls.first_order_critical
            @test sol ≈ xsol atol = 1e-5
            @test A * sol ≈ b atol = 1e-10
            @test all(xlow .- 1e-10 .<= sol .<= xupp .+ 1e-10)
        end
    end
end

# Regression tests on problems with several linear equalities and no bound (HS51 in the
# Hock-Schittkowski collection), and with a starting point that satisfies Ax = b but not
# the bounds (HS21 with the slack variable of its linear inequality)
@testset "Linear equalities and infeasible starting point" begin

    @testset "Several linear equalities" begin
        A = [1.0 3.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 -2.0; 0.0 1.0 0.0 0.0 -1.0]
        b = [4.0, 0.0, 0.0]
        r!(rx, x) = (rx .= [x[1] - x[2], x[2] + x[3] - 2, x[4] - 1, x[5] - 1]; nothing)
        jac_r!(J, x) = (J .= [1.0 -1.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0;
                              0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 1.0]; nothing)
        c!(cx, x) = nothing
        jac_c!(C, x) = nothing
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, fill(-Inf, 5), fill(Inf, 5),
                                   [2.5, 0.5, 2.0, -1.0, 0.5], 5, 4, 0, Val(:only_equalities))
        results = traulls(model)

        @test results.status == Traulls.first_order_critical
        @test results.solution ≈ ones(5) atol = 1e-4
        @test A * results.solution ≈ b atol = 1e-12
    end

    @testset "Starting point outside the bounds" begin
        # min 0.01 x₁² + x₂² s.t. 10x₁ - x₂ - s = 10, s ≥ 0, 2 ≤ x₁ ≤ 50, -50 ≤ x₂ ≤ 50
        A = [10.0 -1.0 -1.0]
        b = [10.0]
        r!(rx, x) = (rx .= [0.1 * x[1], x[2]]; nothing)
        jac_r!(J, x) = (J .= [0.1 0.0 0.0; 0.0 1.0 0.0]; nothing)
        c!(cx, x) = nothing
        jac_c!(C, x) = nothing
        xlow, xupp = [2.0, -50.0, 0.0], [50.0, 50.0, Inf]
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp,
                                   [-1.0, -1.0, -9.0], 3, 2, 0, Val(:only_equalities))
        results = traulls(model)
        sol = results.solution

        @test results.status == Traulls.first_order_critical
        @test sol[1:2] ≈ [2.0, 0.0] atol = 1e-5
        @test A * sol ≈ b atol = 1e-10
        @test all(xlow .- 1e-10 .<= sol .<= xupp .+ 1e-10)
    end
end
