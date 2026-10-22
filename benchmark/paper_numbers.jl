# Aggregated numbers quoted in the text of the paper that the other scripts do not print:
# the comparison of TRAULLS with Ipopt and Percival (Section 4.2), the comparison of
# explicit and penalized linear constraints (Section 4.3) and the comparison of the
# structured and standard secant equations (Section 2.4). They are all computed from the
# files of the results folder, so that no solver is run. The numbers of Section 4.1 are
# printed by make_figures.jl and residual_cost_analysis.jl.
#
# Usage, from the benchmark folder:
#     julia --project=. paper_numbers.jl

using CSV, DataFrames, Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")

# Same relative-gap heuristic as the one used to build the performance profiles
const GAP_TOL = 1e-2

relative_gap(fa, fb) = (max(fa, fb) - min(fa, fb)) / (1 + max(abs(fa), abs(fb)))
geomean(v) = exp(sum(log.(v)) / length(v))
load(file) = CSV.read(joinpath(RESULTS_DIR, file), DataFrame)

function median_of(v)
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

## Section 4.2: TRAULLS (Hybrid-SR1) against Ipopt (L-BFGS) and Percival

traulls = load("traulls_hybrid_sr1.csv")
ipopt = load("ipopt-lbfgs.csv")
percival = load("percival.csv")
ipopt_exact = load("ipopt.csv")

cols(df, tag) = select(df, :name, :status => "status_$tag", :objective => "f_$tag",
                       :elapsed_time => "t_$tag", :neval_residual => "r_$tag",
                       :neval_grad => "g_$tag")
df = innerjoin(cols(traulls, "tr"), cols(ipopt, "ip"), cols(percival, "pe"), on = :name)
df = innerjoin(df, select(ipopt, :name, :nvar), on = :name)
conv_tr = df.status_tr .== "first_order_critical"
conv_ip = df.status_ip .== "first_order"
conv_pe = df.status_pe .== "first_order"

# Instances removed from the profiles: all three solvers converge, to different solutions
fmax = max.(df.f_tr, df.f_ip, df.f_pe)
fmin = min.(df.f_tr, df.f_ip, df.f_pe)
removed = conv_tr .& conv_ip .& conv_pe .& (relative_gap.(fmax, fmin) .> GAP_TOL)
kept = df[.!removed, :]
kconv_tr = kept.status_tr .== "first_order_critical"
kconv_ip = kept.status_ip .== "first_order"
kconv_pe = kept.status_pe .== "first_order"

println("Section 4.2: TRAULLS (Hybrid-SR1), Ipopt (L-BFGS) and Percival")
println("  instances removed from the profiles: ", join(df.name[removed], ", "))
@printf("  instances retained: %d\n", nrow(kept))
for (solver, conv) in (("TRAULLS", kconv_tr), ("Ipopt", kconv_ip), ("Percival", kconv_pe))
    failing = kept.name[.!conv]
    isempty(failing) || println("  $solver does not converge on: ", join(failing, ", "))
end

# Share of the retained instances on which each solver is the fastest (ties count for each)
t = hcat(ifelse.(kconv_tr, kept.t_tr, Inf), ifelse.(kconv_ip, kept.t_ip, Inf),
         ifelse.(kconv_pe, kept.t_pe, Inf))
fastest = t .== minimum(t, dims = 2)
@printf("  fastest solver: TRAULLS %.0f%%, Ipopt %.0f%%, Percival %.0f%% of the instances\n",
        (100 .* vec(sum(fastest, dims = 1)) ./ nrow(kept))...)

small = kept[kept.nvar .<= 100, :]
large = kept[kept.nvar .>= 500, :]
@printf("  TRAULLS faster than Ipopt on %d of the %d instances with n <= 100\n",
        count(small.t_tr .< small.t_ip), nrow(small))
@printf("  TRAULLS faster than Ipopt on %d of the %d instances with n >= 500\n",
        count(large.t_tr .< large.t_ip), nrow(large))
@printf("  time of TRAULLS over that of Ipopt for n >= 500: %.2f (geometric mean)\n",
        geomean(large.t_tr ./ large.t_ip))
@printf("  median times for n <= 100: TRAULLS %.2f ms, Ipopt %.2f ms\n",
        1e3 * median_of(small.t_tr), 1e3 * median_of(small.t_ip))
@printf("  residual evaluations of TRAULLS over those of Ipopt: %.2f (geometric mean)\n",
        geomean(kept.r_tr ./ kept.r_ip))

conv = kept[kconv_pe, :]
@printf("  Percival over TRAULLS on the %d retained instances where Percival converges:\n",
        nrow(conv))
@printf("    time %.2f, residual evaluations %.2f, gradient evaluations %.2f (geometric means)\n",
        geomean(conv.t_pe ./ conv.t_tr), geomean(conv.r_pe ./ conv.r_tr),
        geomean(conv.g_pe ./ conv.g_tr))
timeouts = kept[.!kconv_pe, :]
nrow(timeouts) > 0 &&
    @printf("  time of TRAULLS where Percival fails: at most %.3f s\n", maximum(timeouts.t_tr))

gn = load("traulls_gn.csv")
row(df, name) = df[findfirst(==(name), df.name), :]
h, g = row(traulls, "lv512_1000"), row(gn, "lv512_1000")
@printf("  problem 5.12 with n = 1000: %d inner iterations for Hybrid-SR1 against %d for GN,",
        h.ninner_iter, g.ninner_iter)
@printf(" %.2f s against %.2f s\n", h.elapsed_time, g.elapsed_time)

# NLPModelsIpopt reports the status Restoration_Failed of Ipopt as unknown
exact_failures = ipopt_exact[ipopt_exact.status .!= "first_order", :]
println("  Ipopt with exact second derivatives does not converge on: ",
        join(("$(r.name) ($(r.status))" for r in eachrow(exact_failures)), ", "))

## Section 4.3: linear constraints kept explicit (E) or passed to the augmented Lagrangian (P)

lc = load("linear_constraints.csv")
hyb = lc[lc.hessian .== "hybrid_sr1", :]
E = hyb[(hyb.formulation .== "E") .& (hyb.eps_active .== 1e-3), :]
P = hyb[hyb.formulation .== "P", :]
ep = innerjoin(select(E, :name, :objective => :f_E, :nouter_iter => :outer_E,
                      :ninner_iter => :inner_E, :linear_violation => :viol_E),
               select(P, :name, :objective => :f_P, :nouter_iter => :outer_P,
                      :ninner_iter => :inner_P, :linear_violation => :viol_P), on = :name)
same = relative_gap.(ep.f_E, ep.f_P) .<= GAP_TOL

println("\nSection 4.3: linear constraints, Hybrid-SR1")
@printf("  same solution with both formulations on %d of the %d problems", count(same), nrow(ep))
all(same) || print(" (different on ", join(ep.name[.!same], ", "), ")")
println()
@printf("  outer iterations E/P: %.2f, inner iterations E/P: %.2f (geometric means)\n",
        geomean(ep.outer_E ./ ep.outer_P), geomean(ep.inner_E ./ ep.inner_P))
@printf("  largest violation of the linear constraints: E %.1e, P %.1e\n",
        maximum(ep.viol_E), maximum(ep.viol_P))
println("  solved within a single outer iteration with E: ", join(ep.name[ep.outer_E .== 1], ", "))

## Section 2.4: structured secant equation (31) against the standard one

sc = load("secant_comparison.csv")
println("\nSection 2.4: standard secant equation over structured one, residual evaluations")
println("  (geometric means on the instances where both converge to the same solution; the")
println("  split is on the largest penalty parameter of the run with the structured equation)")
for hess in ("sr1", "bfgs", "hybrid_sr1", "hybrid_bfgs")
    s = sc[(sc.hessian .== hess) .& (sc.secant .== "structured"), :]
    x = sc[(sc.hessian .== hess) .& (sc.secant .== "exact"), :]
    sx = innerjoin(select(s, :name, :status => :st_s, :objective => :f_s,
                          :neval_residual => :r_s, :mu_max),
                   select(x, :name, :status => :st_x, :objective => :f_x,
                          :neval_residual => :r_x), on = :name)
    both = (sx.st_s .== "first_order_critical") .& (sx.st_x .== "first_order_critical") .&
           (relative_gap.(sx.f_s, sx.f_x) .<= GAP_TOL)
    ok = sx[both, :]
    ratio = ok.r_x ./ ok.r_s
    low = ok.mu_max .<= 1e3
    mid = (ok.mu_max .> 1e3) .& (ok.mu_max .<= 1e5)
    @printf("  %-11s %.2f on %d instances (mu <= 1e3: %.2f on %d, 1e3 < mu <= 1e5: %.2f on %d)\n",
            hess, geomean(ratio), nrow(ok), geomean(ratio[low]), count(low),
            geomean(ratio[mid]), count(mid))
    for (label, st) in (("structured", sx.st_s), ("standard", sx.st_x))
        failing = sx.name[st .!= "first_order_critical"]
        isempty(failing) || println("              fails with the $label equation on: ",
                                    join(failing, ", "))
    end
end
println("  median of the difference of the right-hand sides over ||s||^2, by decade of mu:")
for r in eachrow(load("secant_comparison_mu.csv"))
    @printf("    mu = 1e%-3d %10.3g   (share of the term in mu: %.2f, %d updates)\n",
            r.log10_mu, r.median_difference_over_s2, r.median_share_of_mu_Dc, r.nupdates)
end
