# Integration test exercising the SubspaceProjector through the full solver.
# A problem with general linear equality constraints (nlincons > 0) selects the
# SubspaceProjector path in `traulls`; none of the other enabled tests do.
@testset "Solve with linear equality constraints (SubspaceProjector path)" begin

    n = 4
    nres = n
    ncons = 1                       # one nonlinear equality constraint

    target = [1.0, 2.0, 3.0, 4.0]

    # Residuals r(x) = x - target  ⇒  minimize ½‖x - target‖²
    r!(rx, x) = (rx .= x .- target; nothing)
    jac_r!(J, x) = (J .= Matrix{Float64}(I, n, n); nothing)

    # Nonlinear equality constraint x₄ = 0
    c!(cx, x) = (cx[1] = x[4]; nothing)
    jac_c!(C, x) = (C .= [0.0 0.0 0.0 1.0]; nothing)

    # Linear equality constraint  x₁ + x₂ + x₃ + x₄ = 1
    A = reshape([1.0, 1.0, 1.0, 1.0], 1, n)
    b = [1.0]

    xlow = fill(-10.0, n)
    xupp = fill(10.0, n)
    x0 = zeros(n)

    model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, xlow, xupp, x0,
                               n, nres, ncons, Val(:only_equalities))

    @test model.nlincons == 1       # ensures the SubspaceProjector branch is taken

    results = traulls(model; init_mult = false)

    sol = results.solution

    @test results.status isa Traulls.CriticalityStatus
    @test results.feasibility ≤ 1e-5            # nonlinear constraint x₄ = 0
    @test abs(sol[4]) ≤ 1e-5
    @test isapprox(A * sol, b; atol = 1e-6)     # iterate stayed on Ax = b
    @test all(xlow .- 1e-8 .≤ sol .≤ xupp .+ 1e-8)
end
