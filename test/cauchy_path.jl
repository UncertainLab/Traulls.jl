# Cauchy step along the exact projected gradient path s(t) = P_D(-tg) over a polyhedron
# D = {s | As = 0, sₗ ≤ s ≤ sᵤ} (SubspaceProjector). The expected values of the first
# testsets come from a reference implementation that projects onto the critical cones by
# bounded least squares and computes the releases by linear programming.

# Reference projection onto {s | As = b, lo ≤ s ≤ hi} by Dykstra's algorithm
function proj_dykstra(A, lo, hi, w; b = zeros(size(A, 1)), tol = 1e-14, maxit = 200_000)
    Ap = pinv(A)
    Pn = I - Ap * A
    c = Ap * b
    x, p, q = copy(w), zero(w), zero(w)
    for _ in 1:maxit
        y = Pn * (x + p) + c
        p = x + p - y
        xn = clamp.(y + q, lo, hi)
        q = y + q - xn
        norm(xn - x, Inf) <= tol && norm(A * xn - b, Inf) <= 1e-12 && return xn
        x = xn
    end
    return x
end

# Minimum of gᵀs over {s | As = 0, lo ≤ s ≤ hi}, reached at the end of the path
function lp_minimum(A, lo, hi, g)
    n = length(g)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, lo[i] <= v[i = 1:n] <= hi[i])
    @constraint(model, A * v .== 0)
    @objective(model, Min, dot(g, v))
    optimize!(model)
    return objective_value(model)
end

# Cauchy step with H = JᵀJ and its trace, along with the bounds on the step
function cauchy_path(A, xlow, xupp, x, g, Δ; J = zeros(1, length(x)))
    n = length(x)
    P = Traulls.SubspaceProjector(A, cholesky(A * A'))
    H = Traulls.GN(J, zeros(0, n), 0.0)
    slow, supp = max.(-Δ, xlow .- x), min.(Δ, xupp .- x)
    s, d, Hd = zeros(n), zeros(n), zeros(n)
    trace = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]
    pred = Traulls.cauchy_step!(x, s, g, xlow, xupp, slow, supp, H, P, d, Hd; trace = trace)
    return (pred = pred, s = s, trace = trace, slow = slow, supp = supp, P = P)
end

# Breakpoints, directions and number of pieces of a trace, the last entry being the Cauchy
# point
breakpoints(trace) = [e[1] for e in trace[1:end-1]]
directions(trace) = [e[3] for e in trace[1:end-1]]
npieces(trace) = length(trace) - 1

# Point s(t) of the computed path, for t in the range of piece k of the trace
path_point(trace, k, t) = trace[k][2] + (t - trace[k][1]) * trace[k][3]

# Largest relative error between the computed path and the reference projections, at the
# start of the pieces and at `nsample` points inside each of them
function path_error(A, slow, supp, g, trace; nsample = 2)
    err = 0.0
    for k in 1:npieces(trace)
        t0, t1 = trace[k][1], trace[k + 1][1]
        for τ in range(0, 1; length = nsample + 2)[1:end-1]
            t = t0 + τ * (t1 - t0)
            ref = proj_dykstra(A, slow, supp, -t * g)
            err = max(err, norm(path_point(trace, k, t) - ref, Inf) / max(1, norm(ref, Inf)))
        end
    end
    ref = proj_dykstra(A, slow, supp, -trace[end][1] * g)
    return max(err, norm(trace[end][2] - ref, Inf) / max(1, norm(ref, Inf)))
end

# Feasibility of the step and activity of the bounds fixed for the CG iterations
function check_step(A, res)
    s, slow, supp, P = res.s, res.slow, res.supp, res.P
    @test norm(A * s, Inf) <= 1e-10
    @test all(slow .- 1e-12 .<= s .<= supp .+ 1e-12)
    at_bound = [s[i] <= slow[i] + 1e-12 || s[i] >= supp[i] - 1e-12 for i in eachindex(s)]
    @test all(Traulls.is_fixed(P, i) for i in findall(at_bound))
end

# Ω = {x₁ + x₂ + x₃ = 0, x₁ ≥ -1, x₂ ≤ 1}, the example of Section 3.1 of the paper
const A_PAPER = [1.0 1.0 1.0]
const XLOW_PAPER = [-1.0, -Inf, -Inf]
const XUPP_PAPER = [Inf, 1.0, Inf]
const G_PAPER = [0.3, -0.8, 0.5]

# Doubly stochastic 3 × 3 matrices, stored row-major: the three row sums and the first two
# column sums (the third one is implied)
function doubly_stochastic()
    A = zeros(5, 9)
    for r in 1:3, c in 1:3
        A[r, 3(r - 1) + c] = 1
    end
    for c in 1:2, r in 1:3
        A[3 + c, 3(r - 1) + c] = 1
    end
    perms = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    V = [vec(permutedims(Float64.([p[r] == c for r in 1:3, c in 1:3]))) for p in perms]
    return A, V
end

@testset "Cauchy path: example of the paper" begin
    x = [-0.5, -2.0, 2.5]

    # Release of x₁ ≥ -1 after a stationary piece at the vertex (-1, 1, 0), until the trust
    # region bound on s₃ is reached
    res = cauchy_path(A_PAPER, XLOW_PAPER, XUPP_PAPER, x, G_PAPER, 100.0)
    @test breakpoints(res.trace) ≈ [0, 5/3, 55/13, 10, 985]
    @test all(directions(res.trace) .≈ [[-0.3, 0.8, -0.5], [0, 0.65, -0.65], zeros(3),
                                        [0.1, 0, -0.1], zeros(3)])
    @test res.s ≈ [97, 3, -100]
    @test res.pred ≈ -23.3
    check_step(A_PAPER, res)

    # First local minimizer inside the first piece
    res = cauchy_path(A_PAPER, XLOW_PAPER, XUPP_PAPER, x, G_PAPER, 100.0;
                      J = diagm(sqrt.([1.0, 0.5, 2.0])))
    @test res.trace[end][1] ≈ 14/13
    @test res.s ≈ [-0.323077, 0.861538, -0.538462] atol = 1e-6
    @test res.pred ≈ -0.527692 atol = 1e-6

    # The trust region bound on s₂ is reached first
    res = cauchy_path(A_PAPER, XLOW_PAPER, XUPP_PAPER, x, G_PAPER, 2.0)
    @test breakpoints(res.trace) ≈ [0, 5/3, 35/13, 5, 10]
    @test directions(res.trace)[3] == zeros(3) && directions(res.trace)[5] == zeros(3)
    @test directions(res.trace)[4] ≈ [0.1, 0, -0.1]
    @test res.s ≈ [0, 2, -2] atol = 1e-12
    @test res.pred ≈ -2.6
    check_step(A_PAPER, res)
end

@testset "Cauchy path: stationary pieces are skipped" begin
    # Close to the vertex (-1, 1, 0), the decrease is 0.2 + 3.3δ, whereas stopping on the
    # stationary piece at the vertex would yield 3.7δ only, which vanishes with δ
    for δ in (1e-1, 1e-2, 1e-3, 1e-5)
        x = [-1.0, 1.0, 0.0] + δ * [1.0, -3.0, 2.0]
        res = cauchy_path(A_PAPER, XLOW_PAPER, XUPP_PAPER, x, G_PAPER, 1.0)
        @test -res.pred ≈ 0.2 + 3.3δ rtol = 1e-8

        stationary = findfirst(e -> iszero(e[3]), res.trace)
        @test dot(G_PAPER, res.trace[stationary][2]) ≈ -3.7δ rtol = 1e-8
        δ == 1e-3 && @test res.s ≈ [0.997, 0.003, -1.0]
        check_step(A_PAPER, res)
    end
end

@testset "Cauchy path: degenerate vertex of the doubly stochastic polytope" begin
    # At this vertex, nine bounds are active for four degrees of freedom. The point is not
    # critical, while the previous rule, which released the bounds of wrong-sign multipliers
    # and fixed again those along which the direction pointed outwards, gave a zero step
    A, _ = doubly_stochastic()
    x = [1.0, 0, 0, 0, 0, 1, 0, 1, 0]
    g = [-5.0, 5, 0, -6, 8, 8, 7, -4, -3]
    res = cauchy_path(A, zeros(9), ones(9), x, g, 0.5)

    @test npieces(res.trace) == 2
    @test breakpoints(res.trace) ≈ [0, 2/9]
    @test directions(res.trace)[1] ≈ [-2.25, 0, 2.25, 2.25, 0, -2.25, 0, 0, 0] atol = 1e-12
    @test res.s ≈ [-0.5, 0, 0.5, 0.5, 0, -0.5, 0, 0, 0] atol = 1e-12
    @test res.pred ≈ -4.5
    check_step(A, res)
end

@testset "Cauchy path: degenerate polyhedra" begin
    Random.seed!(1)
    A_ds, V_ds = doubly_stochastic()
    polyhedra = (
        ("segment", [1.0 1.0], [[1.0, 0.0], [0.0, 1.0]]),
        ("octahedron", ones(1, 4),
         [Float64.([i in c for i in 1:4]) for c in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
                                                    (3, 4))]),
        ("doubly stochastic", A_ds, V_ds))

    for (name, A, V) in polyhedra
        n = size(A, 2)
        # The vertices and points close to them, on an edge or inside a face
        points = vcat(V, [V[k] + 1e-3 * (V[mod1(k + 1, length(V))] - V[k]) for k in eachindex(V)])
        err, lp_err, pieces = 0.0, 0.0, 0
        for x in points, Δ in (0.05, 0.3, 2.0), gen in (() -> 3 * randn(n),
                                                       () -> Float64.(rand(-3:3, n)))
            g = gen()
            res = cauchy_path(A, zeros(n), ones(n), x, g, Δ)
            check_step(A, res)
            err = max(err, path_error(A, res.slow, res.supp, g, res.trace; nsample = 1))
            lp = lp_minimum(A, res.slow, res.supp, g)
            lp_err = max(lp_err, abs(res.pred - lp) / max(1, abs(lp)))
            pieces = max(pieces, npieces(res.trace))
        end
        @testset "$name" begin
            @test err <= 1e-8
            @test lp_err <= 1e-8
            @test pieces <= 10
        end
    end
end

@testset "Cauchy path: random polyhedra" begin
    Random.seed!(2)
    for (n, m, ninstances, deltas) in ((40, 8, 3, (0.1, 0.5, 2.0)), (100, 20, 1, (0.5,)))
        for _ in 1:ninstances
            # Many active bounds at the projection onto Ω of a random point
            A = randn(m, n)
            b = A * (rand(n) .- 0.5)
            x = proj_dykstra(A, -ones(n), ones(n), 3 * randn(n); b = b)
            for Δ in deltas
                g = randn(n)
                res = cauchy_path(A, -ones(n), ones(n), x, g, Δ)
                check_step(A, res)
                @test path_error(A, res.slow, res.supp, g, res.trace; nsample = 0) <= 1e-8
                lp = lp_minimum(A, res.slow, res.supp, g)
                @test res.pred ≈ lp rtol = 1e-8
                @test npieces(res.trace) <= 2n
            end
        end
    end
end

@testset "Cauchy path: first local minimizer" begin
    Random.seed!(3)
    A_ds, V_ds = doubly_stochastic()
    cases = (
        (A_PAPER, XLOW_PAPER, XUPP_PAPER, [-1.0, 1.0, 0.0] + 1e-2 * [1.0, -3.0, 2.0]),
        (A_PAPER, XLOW_PAPER, XUPP_PAPER, [-0.5, -2.0, 2.5]),
        (A_ds, zeros(9), ones(9), V_ds[1]),
        (A_ds, zeros(9), ones(9), V_ds[4] + 1e-3 * (V_ds[5] - V_ds[4])))

    for (A, xlow, xupp, x) in cases, scale in (0.1, 1.0), Δ in (0.3, 2.0)
        n = length(x)
        g = 3 * randn(n)
        J = scale * randn(n, n)
        q(s) = dot(g, s) + norm(J * s)^2 / 2
        res = cauchy_path(A, xlow, xupp, x, g, Δ; J = J)
        check_step(A, res)
        @test res.pred ≈ q(res.s) atol = 1e-12

        # ψ(t) = q(s(t)) is nonincreasing on [0, t_c] along the reference path ...
        tc = res.trace[end][1]
        ψ = [q(proj_dykstra(A, res.slow, res.supp, -t * g)) for t in range(0, tc; length = 30)]
        @test all(diff(ψ) .<= 1e-10)
        @test ψ[end] ≈ res.pred atol = 1e-10

        # ... and t_c is a local minimizer of ψ
        for h in (1e-6, 1e-4)
            @test q(proj_dykstra(A, res.slow, res.supp, -(tc + h * (1 + tc)) * g)) >=
                  res.pred - 1e-10
        end
    end
end

@testset "Solve from a degenerate vertex of the doubly stochastic polytope" begin
    # Minimize ½‖x - x₀ + g‖² from the vertex x₀ of the testset above, where the gradient
    # is g. The previous rule stalled at x₀, which is not critical
    A, _ = doubly_stochastic()
    x0 = [1.0, 0, 0, 0, 0, 1, 0, 1, 0]
    target = x0 - [-5.0, 5, 0, -6, 8, 8, 7, -4, -3]
    r!(rx, x) = (rx .= x .- target; nothing)
    jac_r!(J, x) = (J .= Matrix{Float64}(I, 9, 9); nothing)
    c!(cx, x) = nothing
    jac_c!(C, x) = nothing

    for eps_active in (1e-3, 0.0)
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, A, ones(5), zeros(9), ones(9),
                                   copy(x0), 9, 9, 0, Val(:only_equalities))
        results = traulls(model; eps_active = eps_active)
        sol = results.solution

        @test results.status == Traulls.first_order_critical
        @test sol ≈ proj_dykstra(A, zeros(9), ones(9), target; b = ones(5)) atol = 1e-6
        @test A * sol ≈ ones(5) atol = 1e-10
        @test all(-1e-10 .<= sol .<= 1 + 1e-10)
    end
end
