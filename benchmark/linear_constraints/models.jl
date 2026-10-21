# Problems with linear constraints under two formulations, for Section 4.3 of the paper:
#   E: linear constraints kept explicit (matrix A of the polyhedron; slack variables are added
#      by hand for the linear inequalities, as in traulls_models/hs22.jl);
#   P: the same constraints passed in c(x), hence handled by the augmented Lagrangian.
# The residuals follow NLSProblems.jl, or traulls_models for the problems of the benchmark.
# LINCONS[name] = (A, lo, hi): linear constraints lo ≤ A x ≤ hi on the original variables,
# used to measure their violation at the final point.

using LinearAlgebra

const LINCONS = Dict{String, Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}}()

# No nonlinear constraint
nocons(n) = (x -> zeros(eltype(x), 0), x -> zeros(0, n))

# Zero columns of the slack variables
pad(M, k) = hcat(M, zeros(size(M, 1), k))

## Problems of the benchmark with explicit linear constraints: HS14, HS22, HS23, HS32, HS323

LINCONS["hs14"] = ([1.0 -2.0], [-1.0], [-1.0])
hs14_E() = hs14()
function hs14_P()
    r(x) = [x[1] - 2.0, x[2] - 1]
    jac_r(x) = [1.0 0.0; 0.0 1.0]
    h(x) = [x[1] - 2x[2] + 1.0]
    jac_h(x) = [1.0 -2.0]
    g(x) = [-0.25 * x[1]^2 - x[2]^2 + 1]
    jac_g(x) = [-0.5*x[1] -2x[2]]
    return Traulls.CnlsModel(r, h, g, jac_r, jac_h, jac_g, fill(-Inf, 2), fill(Inf, 2),
                             2*ones(2), 2, 2, 1, 1)
end

LINCONS["hs22"] = ([-1.0 -1.0], [-2.0], [Inf])
hs22_E() = hs22()
function hs22_P()
    r(x) = [x[1] - 2.0, x[2] - 1.0]
    jac_r(x) = [1.0 0.0; 0.0 1.0]
    g(x) = [-x[1]^2 + x[2], 2.0 - x[1] - x[2]]
    jac_g(x) = [-2x[1] 1.0; -1.0 -1.0]
    return Traulls.CnlsModel(r, g, jac_r, jac_g, fill(-Inf, 2), fill(Inf, 2), [2.0, 2.0],
                             2, 2, 2, Val(:only_inequalities))
end

LINCONS["hs23"] = ([1.0 1.0], [1.0], [Inf])
hs23_E() = hs23()
function hs23_P()
    r(x) = [x[1], x[2]]
    jac_r(x) = [1.0 0.0; 0.0 1.0]
    g(x) = [x[1]^2 + x[2]^2 - 1.0, 9x[1]^2 + x[2]^2 - 9.0, x[1]^2 - x[2], x[2]^2 - x[1],
            x[1] + x[2] - 1.0]
    jac_g(x) = [2x[1] 2x[2]; 18x[1] 2x[2]; 2x[1] -1.0; -1.0 2x[2]; 1.0 1.0]
    return Traulls.CnlsModel(r, g, jac_r, jac_g, [-50.0, -50.0], [50.0, 50.0], [3.0, 1.0],
                             2, 2, 5, Val(:only_inequalities))
end

LINCONS["hs32"] = ([1.0 1.0 1.0], [1.0], [1.0])
hs32_E() = hs32()
function hs32_P()
    r(x) = [x[1] + 3x[2] + x[3], 2(x[1] - x[2])]
    jac_r(x) = [1.0 3.0 1.0; 2.0 -2.0 0.0]
    h(x) = [x[1] + x[2] + x[3] - 1.0]
    jac_h(x) = [1.0 1.0 1.0]
    g(x) = [6x[2] + 4x[3] - x[1]^3 - 3.0]
    jac_g(x) = [-3x[1]^2 6.0 4.0]
    return Traulls.CnlsModel(r, h, g, jac_r, jac_h, jac_g, zeros(3), fill(Inf, 3),
                             [0.1, 0.7, 0.2], 3, 2, 1, 1)
end

LINCONS["hs323"] = ([1.0 -1.0], [-2.0], [Inf])
hs323_E() = hs323()
function hs323_P()
    r(x) = [x[1] - 2.0, x[2]]
    jac_r(x) = [1.0 0.0; 0.0 1.0]
    g(x) = [-x[1]^2 + x[2] - 1.0, x[1] - x[2] + 2.0]
    jac_g(x) = [-2x[1] 1.0; 1.0 -1.0]
    return Traulls.CnlsModel(r, g, jac_r, jac_g, zeros(2), fill(Inf, 2), [0.0, 1.0],
                             2, 2, 2, Val(:only_inequalities))
end

## Problems of the benchmark whose linear constraints are in c(x): HS42, HS49, HS50

LINCONS["hs42"] = ([1.0 0.0 0.0 0.0], [2.0], [2.0])
hs42_P() = hs42()
function hs42_E()
    r(x) = [x[1] - 1.0, x[2] - 2.0, x[3] - 3.0, x[4] - 4.0]
    jac_r(x) = Matrix(1.0I, 4, 4)
    c(x) = [x[3]^2 + x[4]^2 - 2]
    jac_c(x) = [0.0 0.0 2x[3] 2x[4]]
    return Traulls.CnlsModel(r, c, jac_r, jac_c, [1.0 0.0 0.0 0.0], [2.0], fill(-Inf, 4),
                             fill(Inf, 4), ones(4), 4, 4, 1, Val(:only_equalities))
end

const A49 = [1.0 1.0 1.0 4.0 0.0; 0.0 0.0 1.0 0.0 5.0]
LINCONS["hs49"] = (A49, [7.0, 6.0], [7.0, 6.0])
hs49_P() = hs49()
function hs49_E()
    r(x) = [x[1] - x[2], x[3] - 1.0, (x[4] - 1.0)^2, (x[5] - 1.0)^3]
    jac_r(x) = [1.0 -1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 2*(x[4]-1.0) 0.0; 0.0 0.0 0.0 0.0 3*(x[5]-1.0)^2]
    c, jac_c = nocons(5)
    return Traulls.CnlsModel(r, c, jac_r, jac_c, A49, [7.0, 6.0], fill(-Inf, 5), fill(Inf, 5),
                             [10.0, 7.0, 2.0, -3.0, 0.8], 5, 4, 0, Val(:only_equalities))
end

const A50 = [1.0 2.0 3.0 0.0 0.0; 0.0 1.0 2.0 3.0 0.0; 0.0 0.0 1.0 2.0 3.0]
LINCONS["hs50"] = (A50, fill(6.0, 3), fill(6.0, 3))
hs50_P() = hs50()
function hs50_E()
    r(x) = [x[1] - x[2], x[2] - x[3], (x[3] - x[4])^2, x[4] - x[5]]
    jac_r(x) = [1.0 -1.0 0.0 0.0 0.0; 0.0 1.0 -1.0 0.0 0.0;
                0.0 0.0 2*(x[3] - x[4]) 2*(x[4]-x[3]) 0.0; 0.0 0.0 0.0 1.0 -1.0]
    c, jac_c = nocons(5)
    return Traulls.CnlsModel(r, c, jac_r, jac_c, A50, fill(6.0, 3), fill(-Inf, 5), fill(Inf, 5),
                             [-35.0, -31.0, 11.0, 5.0, -5.0], 5, 4, 0, Val(:only_equalities))
end

## Other problems, from NLSProblems.jl

# Model with linear equality constraints only, under both formulations
function linear_equalities(r, jac_r, A, b, low, upp, x0, nres, form)
    n = length(x0)
    if form == :E
        c, jac_c = nocons(n)
        return Traulls.CnlsModel(r, c, jac_r, jac_c, A, b, low, upp, x0, n, nres, 0,
                                 Val(:only_equalities))
    else
        return Traulls.CnlsModel(r, x -> A*x - b, jac_r, x -> A, low, upp, x0, n, nres,
                                 size(A, 1), Val(:only_equalities))
    end
end

# Model with linear inequalities lo ≤ A x ≤ hi only. Formulation E adds one slack variable
# s = A x bounded by [lo, hi] per row; formulation P puts the finite inequalities in c(x) ≥ 0
function linear_inequalities(r, jac_r, A, lo, hi, low, upp, x0, nres, form)
    nx, k = length(x0), size(A, 1)
    if form == :E
        n = nx + k
        c, jac_c = nocons(n)
        rs(x) = r(x[1:nx])
        jac_rs(x) = pad(jac_r(x[1:nx]), k)
        return Traulls.CnlsModel(rs, c, jac_rs, jac_c, hcat(A, -Matrix(1.0I, k, k)), zeros(k),
                                 vcat(low, lo), vcat(upp, hi), vcat(x0, A*x0), n, nres, 0,
                                 Val(:only_equalities))
    else
        rows_lo = findall(isfinite, lo); rows_hi = findall(isfinite, hi)
        G = vcat(A[rows_lo, :], -A[rows_hi, :])
        d = vcat(lo[rows_lo], -hi[rows_hi])
        return Traulls.CnlsModel(r, x -> G*x - d, jac_r, x -> G, low, upp, x0, nx, nres,
                                 size(G, 1), Val(:only_inequalities))
    end
end

const A48 = [1.0 1.0 1.0 1.0 1.0; 0.0 0.0 1.0 -2.0 -2.0]
LINCONS["hs48"] = (A48, [5.0, -3.0], [5.0, -3.0])
for form in (:E, :P)
    @eval $(Symbol("hs48_", form))() = linear_equalities(
        x -> [x[1] - 1.0, x[2] - x[3], x[4] - x[5]],
        x -> [1.0 0.0 0.0 0.0 0.0; 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 1.0 -1.0],
        A48, [5.0, -3.0], fill(-Inf, 5), fill(Inf, 5), [3.0, 5.0, -3.0, 2.0, -2.0], 3, $(QuoteNode(form)))
end

const A51 = [1.0 3.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 -2.0; 0.0 1.0 0.0 0.0 -1.0]
r51(x) = [x[1] - x[2], x[2] + x[3] - 2, x[4] - 1, x[5] - 1]
jac51(x) = [1.0 -1.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 1.0]
LINCONS["hs51"] = (A51, [4.0, 0.0, 0.0], [4.0, 0.0, 0.0])
for form in (:E, :P)
    @eval $(Symbol("hs51_", form))() = linear_equalities(r51, jac51, A51, [4.0, 0.0, 0.0],
        fill(-Inf, 5), fill(Inf, 5), [2.5, 0.5, 2.0, -1.0, 0.5], 4, $(QuoteNode(form)))
end

const A52 = [1.0 3.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 -2.0; 0.0 1.0 0.0 0.0 -1.0]
r52(x) = [4x[1] - x[2], x[2] + x[3] - 2, x[4] - 1, x[5] - 1]
jac52(x) = [4.0 -1.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 1.0]
LINCONS["hs52"] = (A52, zeros(3), zeros(3))
for form in (:E, :P)
    @eval $(Symbol("hs52_", form))() = linear_equalities(r52, jac52, A52, zeros(3),
        fill(-Inf, 5), fill(Inf, 5), fill(2.0, 5), 4, $(QuoteNode(form)))
end

LINCONS["hs53"] = (A52, zeros(3), zeros(3))
for form in (:E, :P)
    @eval $(Symbol("hs53_", form))() = linear_equalities(r51, jac51, A52, zeros(3),
        fill(-10.0, 5), fill(10.0, 5), fill(2.0, 5), 4, $(QuoteNode(form)))
end

LINCONS["hs21"] = ([10.0 -1.0], [10.0], [Inf])
for form in (:E, :P)
    @eval $(Symbol("hs21_", form))() = linear_inequalities(
        x -> [0.1 * x[1], x[2]], x -> [0.1 0.0; 0.0 1.0], [10.0 -1.0], [10.0], [Inf],
        [2.0, -50.0], [50.0, 50.0], [-1.0, -1.0], 2, $(QuoteNode(form)))
end

const A224 = [1.0 3.0; 1.0 1.0]
LINCONS["tp224"] = (A224, [0.0, 0.0], [18.0, 8.0])
for form in (:E, :P)
    @eval $(Symbol("tp224_", form))() = linear_inequalities(
        x -> [sqrt(2) * (x[1] - 12), x[2] - 20], x -> [sqrt(2) 0.0; 0.0 1.0],
        A224, [0.0, 0.0], [18.0, 8.0], [0.0, 0.0], [6.0, 6.0], [0.1, 0.1], 2, $(QuoteNode(form)))
end

const D268 = [-74.0 80 18 -11 -4; 14 -69 21 28 0; 66 -72 -5 7 1; -12 66 -30 -23 3;
              3 8 -7 -4 1; 4 -12 4 4 0]
const d268 = [51.0, -61, -56, 69, 10, -12]
const A268 = [-1.0 -1 -1 -1 -1; 10 10 -3 5 4; -8 1 -2 -5 3; 8 -1 2 5 -3; -4 -2 3 -5 1]
const c268 = [5.0, -20, 40, -11, 30]
LINCONS["tp268"] = (A268, -c268, fill(Inf, 5))
for form in (:E, :P)
    @eval $(Symbol("tp268_", form))() = linear_inequalities(
        x -> D268 * x[1:5] - d268, x -> D268, A268, -c268, fill(Inf, 5),
        fill(-Inf, 5), fill(Inf, 5), ones(5), 6, $(QuoteNode(form)))
end

const LIN_PROBLEMS = ["hs14", "hs21", "hs22", "hs23", "hs32", "hs42", "hs48", "hs49", "hs50",
                      "hs51", "hs52", "hs53", "hs323", "tp224", "tp268"]
