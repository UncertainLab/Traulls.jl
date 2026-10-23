# Cauchy step along the exact projected gradient path — implementation plan

*Traulls.jl, COAP revision (`preprint/coap/sn-preprint-theorie.tex`). Written 2026-10-23.*

This document specifies a new Cauchy step for problems with linear equality constraints
(`SubspaceProjector` path). Over a general polyhedron, it computes the first local minimizer
of the quadratic model along the **exact** projected gradient path
$s(t) = P_D(-tg)$. The former nested-working-set path is not that path. The document
covers the mathematics, the algorithm, the code changes file by file, the tests (with
expected values), and the changes to the paper. A Python reference implementation, checked
against QP projections, is in Appendix A.

The Cauchy step is implemented in `src/cauchy.jl` and tested in `test/cauchy_path.jl`.
Sections 3 to 6 describe the implementation; §7 (benchmarks) and §8 (paper) are still to do.

---

## 0. Summary

- On each piece, the projected gradient path moves along $-P_T g$, where $T$ is the null
  space of $A$ and of the active bounds. `cauchy_step!` already computes this direction.
  What the former code lacked:
  1. **Releases.** An active bound leaves the active set when its multiplier crosses zero.
     On a piece, the multipliers $\eta(t)$ of the projection are affine in $t$, and their
     slope $\theta$ is `bound_multipliers(proj_op)` right after projecting $g$. So the
     release time comes from an $O(p)$ ratio test. The released bound is then freed by the
     local cone projection (item 2) with `set_free!` (rank-one update plus
     `append_redundant!`), which already existed.
  2. **An exact working set** at $t=0$ and at every breakpoint. It comes from a small QP
     (the projection of $-g$ onto a critical cone), solved by a primal active-set loop that
     uses `set_active!` / `set_free!`. At a nondegenerate breakpoint this loop changes the
     working set at most once.
  3. **Stationary pieces** (the path waits at a vertex until a release). They are skipped
     when looking for the first local minimizer, and are not a stopping condition.
- **Box-only problems** (`CoordinateSubspaceProjector`) are unchanged. On a box, no bound is
  ever released, and the former routine, which is kept, already computes the exact path.
- **The ε-active set** is no longer used in the Cauchy step (ε = 0 there). It stays only in
  the stopping test of the inner loop (the Conn–Gould–Sartenaer–Toint 1996 test), with
  **any** $\varepsilon_{\max}>0$. The threshold $\bar\varepsilon$ of Lemma
  `lemma:eps_threshold` disappears, because only the nested path needed it (see §8,
  "What ε must satisfy").
- **In the paper**, Assumption `assumption:full_rank_active`, Lemma `lemma:eps_threshold`,
  Proposition `prop:computed_path`, Lemma `lemma:initial_direction`, Theorem
  `theo:cauchy_decrease` and the zigzag remark are replaced by the standard decrease theory
  for the projected gradient path, which needs no linear independence assumption, plus one
  short lemma linking the inner measure to the ε-test. The outer analysis does not change.

---

## 1. Why

The former procedure (`initial_fixed!` plus the main loop of `cauchy_step!`), still
described in the paper (§2.3, §3.1 and Appendix A), worked as follows: fix the ε-active
bounds, free those whose multiplier has the wrong sign, fix again those along which the
direction points outwards, then follow nested working sets (a bound is never released). Its decrease guarantee (Lemma
`lemma:initial_direction`, Theorem `theo:cauchy_decrease`) relies on Assumption
`assumption:full_rank_active`: for every $x\in\Omega$, $\tilde A = [A; E_{\mathcal A(x)}^\top]$
has full row rank. That assumption means every vertex of $\Omega$ is nondegenerate in the
linear-programming sense:

- It holds automatically for boxes and for almost every right-hand side $b$.
- It fails for $\{x_1+x_2=1,\ x\in[0,1]^2\}$ (both vertices), for any simplex with explicit
  upper bounds, for the octahedron $\{\sum_i x_i = 2,\ x\in[0,1]^4\}$, and for
  transportation and assignment polytopes. In the last two cases no bound is redundant, so
  removing redundant bounds in preprocessing does not help.

Without the assumption, the **algorithm** fails, not only the proof:

- **With minimum-norm least-squares multipliers**, on $x_1+x_2=1$, $x\in[0,1]^2$, at
  $x=(0,1)$ with $g=(-2.83,-2.31)$: the point is not critical, yet $V=\{x_1\ge 0\}$, $T_F=\{0\}$
  and $z_0=0$. The inner loop never terminates.
- **With minimum-sign-violation multipliers** (what the code does), on the 3×3 doubly
  stochastic polytope. Store the 3×3 matrix row-major as a 9-vector. Take $A$ = the 3 row
  sums and the first 2 column sums, $l=0$, $u=1$, the vertex $x=(1,0,0;\,0,0,1;\,0,1,0)$ and
  $g=[-5,5,0,-6,8,8,7,-4,-3]$. The point is not critical (an LP finds a feasible
  descent direction with $g^\top d=-9$), yet the release/re-fix rule ends with $z_0=0$ (with
  ε = 0 or 1e-3, and with either multiplier choice). The exact path moves along
  $(-2.25,0,2.25;\,2.25,0,-2.25;\,0,0,0)$, and with $\Delta=0.5$ and $H=0$ the decrease is
  $4.5$. This comes from a Python re-implementation of eq. `(refixing)`. The former Julia
  code reproduced the stall: `pred = 0` and $s=0$. A solve of
  $\min\tfrac12\|x-x_0+g\|^2$ from this vertex $x_0$ ended `feasible_non_critical` after 600
  inner iterations without leaving it.

Following the exact path removes the need for the assumption and for ε in the Cauchy step.
The standard theory of the projected gradient path on convex sets then applies.

---

## 2. Mathematical description

### 2.1 Setting

At inner iterate $x\in\Omega$ with gradient $g$, model $q(s)=\tfrac12 s^\top Hs+g^\top s$ and
radius $\Delta$, the step set is

$$D=\{s \mid As=0,\ s_l\le s\le s_u\},\qquad s_l=\max(-\Delta,\,l-x),\quad s_u=\min(\Delta,\,u-x),$$

so $0\in D$, and trust-region bounds are ordinary bounds (`slow`, `supp` in the code). The
path is $s(t)=P_D(-tg)$, $t\ge 0$. For an active bound on variable $i$, write
$\sigma_i=+1$ at a lower bound, $\sigma_i=-1$ at an upper bound, and $\sigma_i=0$ for a
fixed variable ($s_{l,i}=s_{u,i}$). The inward normal is $\sigma_i e_i$.

### 2.2 Shape of the path

$s(t)$ is continuous and piecewise linear. On a piece where the working set (the bounds
held at their value) is $\mathcal W$,

$$s'(t)=-z,\qquad z=P_{T(\mathcal W)}\,g,\qquad T(\mathcal W)=\{v\mid Av=0,\ v_i=0\ (i\in\mathcal W)\},$$

which is what `mul!(d, proj_op, g, -1, 0)` returns ($d=-z$).

### 2.3 Multipliers along a piece

The KKT conditions of $\min_s \tfrac12\|s+tg\|^2$ over $D$ read

$$tg+s(t)=A^\top\xi(t)+\sum_{i\in\mathcal W}\eta_i(t)\,e_i,\qquad \sigma_i\eta_i(t)\ge 0 .$$

Differentiating along a piece, with $s'=-z$ and $z\in T$, gives
$g-z=A^\top\xi'+E_{\mathcal W}\eta'$. So $\eta'=\theta$, the bound part of the least-squares
decomposition $g=P_Tg+A^\top y_1+E_{\mathcal W}y_2$:

$$\eta(t)=\eta(t_i)+(t-t_i)\,\theta,\qquad \eta(0)=0 .$$

- **In the code**, `bound_multipliers(proj_op)` (a view on `pbuf`, indexed by
  `k` ↔ `fixidx[k]`) holds $\theta$ right after `mul!(d, proj_op, g, -1, 0)`. The scaling
  `-1` applies to the output only, so these are the multipliers of $+g$. Any other `mul!`
  with `proj_op` overwrites them: copy $\theta$ into an `n`-indexed buffer immediately.
- **Sign convention**, the same as `wrong_sign_bounds`: the right sign is $\theta,\eta\ge0$
  at a lower bound and $\le 0$ at an upper bound.
- **Redundant bounds** (flagged in `fixvars` but not in the factor, `fixpos[i] == 0`) have
  no multiplier in this representation. Keep their $\eta_i=0$. The accumulated $\eta$
  remains a valid multiplier vector: right signs, and it satisfies the KKT equation.

### 2.4 Events (breakpoints)

1. **Hit.** A free variable reaches $s_{l,i}$ or $s_{u,i}$. This is `next_breakpoint`,
   unchanged.
2. **Release.** An active, represented, non-fixed bound $j$ with $\sigma_j\theta_j<0$
   reaches $\eta_j=0$ at
   $$t=t_i+\frac{\sigma_j\eta_j(t_i)}{-\sigma_j\theta_j}.$$
   The implementation requires $\sigma_j\theta_j<-\sqrt\epsilon\,\|g\|_\infty$, the noise
   floor of the projection, so that slopes that are zero in exact arithmetic do not
   schedule releases at $t\approx10^{14}$ (§3.1).

The next breakpoint is the earlier of the two kinds. Ties count as one event: event times
within `epsbp` are merged, as before, and so are hits within the rounding errors of $s$,
$O(\epsilon\,t\|g\|_\infty)$, which `epsbp` misses (§3.1).

### 2.5 Direction at $t=0$ and at a breakpoint: local cone projection

The right derivative of the path at a point $(t,s)$ is the projection of $-g$ onto the
critical cone

$$K=T_D(s)\cap\{d \mid \langle d,\ tg+s\rangle=0\}$$

(Haraux 1977, directional derivative of the projection onto a polyhedron). Here
$T_D(s)=\{d\mid Ad=0,\ \sigma_i d_i\ge0 \text{ for bounds active at } s\}$. For **any**
valid multiplier vector $\eta$:

$$K=\{d\in T_D(s)\mid d_i=0\ \text{whenever } \eta_i\neq 0\}.$$

*Proof.* For $d$ with $Ad=0$, $\langle d,tg+s\rangle=\sum_i\eta_i d_i$, and each term
$\eta_i d_i=(\sigma_i\eta_i)(\sigma_i d_i)\ge0$ on $T_D(s)$. So the sum vanishes iff every
term vanishes. ∎

So the accumulated $\eta$ classifies the bounds active at $s$:

- **strongly active** $S=\{i \mid |\eta_i|>\text{tol}_\eta\}$: equality $d_i=0$;
- **weakly active** $W=\{i \mid \eta_i\approx 0\}$: sign constraint $\sigma_i d_i\ge 0$.
  This set contains the bounds just hit, the bounds just released, the redundant bounds,
  and the bounds that stayed at their value while free on the previous piece (parked,
  §3.1). The latter are active at $s$ with a zero multiplier, so they must be in $W$
  although they are not in the working set.

The local problem is

$$\min_d\ \tfrac12\|d+g\|^2\quad\text{s.t.}\quad Ad=0,\quad d_i=0\ (i\in S),\quad \sigma_i d_i\ge0\ (i\in W).$$

Its KKT conditions, for a working set $\mathcal W$ with $S\subseteq\mathcal W\subseteq S\cup W$
and $d=-P_{T(\mathcal W)}g$, are:

- primal: $\sigma_i d_i\ge 0$ for $i\in W\setminus\mathcal W$;
- dual: $\sigma_i\theta_i\ge0$ for $i\in W\cap\mathcal W$.

At $t=0$: $S=\emptyset$, and $W$ is every bound active at $x$ (the TR bounds are not

active at $s=0$). The solution is $P_{T_\Omega(x)}(-g)$. The former
`initial_fixed!(::SubspaceProjector, …)` was only the first iteration of the active-set
method below.

**Nondegenerate breakpoints need at most one change of the working set.** For a single
hit of bound $j$, the unconstrained continuation is the previous direction, which points
outwards. So $j$ must be in the working set and its $\theta_j$ has the right sign. For a
single release of $j$, fixing $j$ gives a multiplier of the wrong sign beyond $t$, so $j$
is free and $d_j$ points inwards. The implementation starts the active-set loop with every
weakly active bound fixed, hit and released bounds included, since that start is always
feasible. At a single hit, the loop ends without changing the working set; at a single
release, it frees $j$ and takes one full step. It does real work only at degenerate
breakpoints: ties, or several weakly active bounds with dependent rows.

### 2.6 Stationary pieces and the end of the path

- If $d=0$, the path is stationary: $s$ does not move while $\eta$ keeps evolving. If a
  release is ahead, jump to it ($s$ and `pred` unchanged) and continue. The paper's §3.1
  example is stationary at the vertex $(-1,1,0)$ for $t\in[55/13,\,10]$.
- If $d=0$ and no release is ahead, i.e. all represented active bounds satisfy
  $\sigma_j\theta_j\ge0$ (in the implementation, $\ge-\sqrt\epsilon\,\|g\|_\infty$), the
  path has ended: $s(t)$ is constant from then on.
- With a finite TR box, $D$ is bounded, so a ray with $d\ne0$ always meets a bound.

### 2.7 First local minimizer of $\psi(t)=q(s(t))$

Same formulas as before on each piece with $d\neq0$: $\psi'=g^\top d+s^\top Hd$, $\psi''=d^\top Hd$.

- Stop at the start of the piece if $\psi'\ge0$ (with the former `eps_slope` / `eps_curv`
  safeguard).
- Stop inside the piece at $-\psi'/\psi''$ if $\psi''>0$ and that point lies before the
  next event.
- Otherwise, advance to the next event.
- **Pieces with $d=0$ are skipped. They are not a stopping condition.** Stopping on them
  is exactly the zigzag failure: with $x=(-1,1,0)+\delta(1,-3,2)$, $H=0$ and $\Delta=1$,
  stopping on the stationary piece gives a decrease of $3.7\delta$, whereas the rule above
  gives $0.2+3.3\delta$ (0.53 / 0.233 / 0.2033 / 0.200033 for $\delta$ = 1e-1 / 1e-2 / 1e-3 /
  1e-5), while $\|P_\Omega(x-g)-x\|\to 0.1414$.
- The former stop on a tiny direction (`norm(d, Inf) <= sqrt(eps)·norm(g, Inf)` after the
  first piece, `10 eps` on the first one) has become: set $d=0$ and treat the piece as
  stationary, with the same thresholds.

### 2.8 Properties

- **Finite.** Each working set occurs on at most one interval of $t$ (critical regions of a
  one-parameter strictly convex parametric QP are intervals). The number of pieces is no
  longer bounded by $n-m$, and in the worst case it can be exponential (as for the Lasso
  path, Mairal & Yu, ICML 2012). In tests it stays around $0.75\,n$, and the Cauchy point
  is reached after about $0.4\,n$ pieces (Appendix B). Keep a safeguard on the number of
  pieces anyway.
- **Box case.** $s_i(t)=\operatorname{clamp}(-tg_i)$ is monotone, so nothing is ever
  released, and the procedure reduces to the `CoordinateSubspaceProjector` path, which is
  kept.
- **Cost per piece.** As before: one projection and one Hessian-vector product, plus an
  $O(p)$ ratio test for releases, $O(p^2)$ per release (`remove_active!`), one projection
  per further change of the working set in the cone projection, and $O(n)$ for the tie
  test and the weak set.

---

## 3. Algorithm

This section describes the implementation in `src/cauchy.jl`. Besides `eps_slope` and
`eps_curv` (absolute, $10^{-10}$, as in the box case), it uses the following tolerances.
Rounding errors in the path are relative to $\|g\|_\infty$ for directions and multipliers,
and to the scale $t\|g\|_\infty$ of the projected point for the step.

| Name | Value | Use |
|---|---|---|
| `tol_sign` | $10\,\epsilon\,\|g\|_\infty$ | sign of the multipliers in the cone projection; parking |
| `tol_noise` | $\sqrt\epsilon\,\|g\|_\infty$ | noise floor of the projection: steps of the cone projection, slopes of the releases |
| `tol_dir` | $10\,\epsilon\,\|g\|_\infty$ on the first piece, $\sqrt\epsilon\,\|g\|_\infty$ after | stationary piece |
| `epsbp` | $10\,\epsilon$ | ties between event times |
| `tol_tie` | $100\,\epsilon\,t\,\|g\|_\infty$ | ties between hits, measured on $s$ |
| `tol_η` | $100\,\epsilon\,\|tg+s\|_\infty$ | strongly versus weakly active |

### 3.1 Main loop (`cauchy_step!`, `SubspaceProjector` method)

```
Input : x, g, H, ℓ, u, slow, supp (TR included), proj_op, buffers d, Hd,
        η, θ, d̂, p ∈ ℝⁿ, σ ∈ Int8ⁿ (from the Workspace, via cauchy_buffers),
        maxpieces = 10(n−m) + 10, trace (optional)
Output: s (Cauchy step), pred = q(s); proj_op encodes A and every bound active at x + s

# bounds active at x, snapped so that their activity is exact along the path
reset_projector!(proj_op) ; W ← ∅
for i = 1:n                                      # same test as identify_active_set!
    lo ← near_lower_bound(x, ℓ, i, √ε, 0) ; up ← near_upper_bound(x, u, i, √ε, 0)
    lo ⇒ slow_i ← 0 ;  up ⇒ supp_i ← 0
    lo or up ⇒ σ_i ← (lo and up ? 0 : lo ? +1 : −1) ; W ← W ∪ {i}
s ← 0 ; η ← 0 ; t ← 0 ; pred ← 0 ; inside ← false
local_cone_projection!(proj_op, g, d, θ, W, σ)               # d = P_{T_Ω(x)}(−g)

for piece = 1, 2, …
    # stationary piece
    stationary ← ‖d‖∞ ≤ tol_dir ;  stationary ⇒ d ← 0
    # parked bounds: free weakly active bounds along which d vanishes
    Park ← {j ∈ W free : σ_j d_j ≤ tol_sign} ;  d_j ← 0 (j ∈ Park)

    # next event (§2.4), ties within epsbp
    (Δhit, Ihit) ← stationary ? (∞, ∅) : next_breakpoint(d, s, slow, supp, proj_op)
    (Δrel, Irel) ← next_release(η, θ, σ, proj_op; tol = tol_noise)
    Δ ← min(Δhit, Δrel) ;  Δhit − Δ ≥ epsbp ⇒ Ihit ← ∅ ;  Δrel − Δ ≥ epsbp ⇒ Irel ← ∅
    trace: push (t, s, d)

    # first local minimizer, skipping stationary pieces
    if not stationary
        Hd ← H d ; φ1 ← gᵀd + sᵀHd ; φ2 ← dᵀHd
        φ1 ≥ 0 or (|φ1| ≤ eps_slope and φ2 > eps_curv) ⇒ break         # s is the Cauchy point
        if φ2 > 0 and −φ1/φ2 < Δ                                        # minimizer inside the piece
            τ ← −φ1/φ2 ; s ← s + τd ; pred += φ1τ + φ2τ²/2 ; t += τ ; inside ← true ; break
    Δ == ∞ ⇒ break                                                      # end of the path
    piece == maxpieces ⇒ warn, break

    # move to the breakpoint
    if not stationary
        s ← s + Δd ; pred += φ1Δ + φ2Δ²/2
        Ihit ← Ihit ∪ {i free : (d_i < 0 and s_i − slow_i ≤ tol_tie)
                             or (d_i > 0 and supp_i − s_i ≤ tol_tie)}   # tol_tie at t + Δ
    η_j ← η_j + Δθ_j for every represented bound j ;  t ← t + Δ
    i ∈ Ihit :  (s_i, σ_i) ← (slow_i, +1) if d_i < 0, (supp_i, −1) otherwise ;  η_i ← 0
    i ∈ Irel :  η_i ← 0                                   # still fixed, now weakly active

    # weakly active bounds at the breakpoint, then direction of the next piece
    W ← Ihit ∪ Park ∪ {i fixed : |η_i| ≤ tol_η} ;  η_i ← 0 on the last set
    local_cone_projection!(proj_op, g, d, θ, W, σ)
end
trace: push (t, s, 0)
# every bound active at x + s fixed for CG
set_active!(proj_op, inside ? Park : W) ;  update_inner_active_set!(s, slow, supp, proj_op)
return pred
```

The last set in the update of `W` contains the released bounds and the redundant bounds,
whose $\eta$ is 0.

Notes:
- **Releases are not applied in the main loop.** A released bound only gets $\eta_i=0$ and
  joins `W`, still fixed. `local_cone_projection!` starts with every bound of `W` fixed and
  frees the released bound if its multiplier has the wrong sign. Freeing it beforehand (the
  natural guess hit → fixed, released → free, of the first version of this plan) would be
  undone by that start, and it is not a feasible start at a degenerate release.
- **`next_release`** loops over `k in 1:proj_op.p`, with `i = fixidx[k]`, skips
  `σ_i == 0`, and keeps $\sigma_i\theta_i<-\texttt{tol\_noise}$. The gap is
  $\max(\sigma_i\eta_i,0)/(-\sigma_i\theta_i)$, so it is 0 if $\sigma_i\eta_i<0$ by
  rounding.
  - The threshold is the noise floor, not `tol_sign`: slopes that are zero in exact
    arithmetic come out around $-10^{-14}$, and with `tol_sign` they scheduled fake releases
    at $t\approx6\cdot10^{13}$ on the doubly stochastic polytope.
  - A genuine release whose slope is below the floor would occur at
    $t\gtrsim t_i/\sqrt\epsilon$. It is ignored, so the path is truncated there.
- **Parked bounds.** The cone projection may leave free a weakly active bound $j$ along
  which $d_j$ vanishes (dependent rows, or a zero multiplier with a zero direction).
  Outward components up to `tol_noise` are rounding errors that the cone projection
  ignores (§3.2).
  - Without parking, an outward $d_j$ of $10^{-17}$ would give a zero-length hit, and the
    same cone problem would free $j$ again: an endless sequence of zero-length pieces.
  - With $d_j=0$, $j$ stays exactly at its bound along the piece and joins `W` at the next
    breakpoint. $Ad$ changes by at most `tol_noise`.
  - On a stationary piece, every free bound of `W` is parked.
- **Ties measured on $s$.** `epsbp` compares event times, which does not scale with the
  problem: two hits that are simultaneous in exact arithmetic were found $3\cdot10^{-15}$
  apart in $t$, which left a bound $10^{-15}$ from its value and not fixed.
  - The rounding errors on $s$ are $O(\epsilon\,t\|g\|_\infty)$. After the move, a free
    variable within `tol_tie` of the bound it moves towards is also hit, and snapped.
  - Only variables moving towards a bound, at a breakpoint, are tested. The active set
    stays combinatorial: there is no re-detection of activity from $s$ (§5).
- **Fixed components of $d$ are not zeroed**, so that $As=0$ holds to machine precision.
  The steps of the fixed variables drift by $O(\epsilon\,t\|g\|_\infty)$.
- **Bounds fixed for CG.**
  - If $s^C$ is a breakpoint (stop at the start of a piece, end of the path, `maxpieces`),
    every bound of `W` is at its value, including those just freed by the cone projection.
    If $s^C$ is inside a piece, only the parked ones are.
  - They are fixed combinatorially. `update_inner_active_set!` misses them: its tolerance
    is relative to $|s_{l,i}|$, which is 0 for the bounds snapped at the start, and their
    drift may point inwards. It then only catches the bounds reached by coincidence.
- $\eta$ and $\theta$ are stored **per variable** (`n`-vectors), because `fixidx` is
  reordered by `remove_active!`. `projected_direction!` scatters $\theta$
  (`copy_bound_multipliers!`) right after each projection. Redundant bounds have
  $\theta_i=0$ and keep $\eta_i=0$.
- **Warnings** are issued if `maxpieces` is reached, in which case $s$ stays at the start
  of the current piece (feasible, with $\psi$ nonincreasing so far), and at the end if some
  call of `local_cone_projection!` did not converge. Neither occurred in the tests.
- `trace` receives $(t, s(t), d)$ at the start of each piece examined ($d=0$ on a
  stationary piece), then $(t_C, s^C, 0)$.
- Not implemented: the debug check $\|P(tg+s)\|\approx0$ at breakpoints.

### 3.2 `local_cone_projection!` — primal active set (Nocedal & Wright, Alg. 16.3)

Problem of §2.5, with the strongly active bounds $S$ (fixed in `proj_op` on entry, always
kept in the working set) and the weakly active ones $W$. Below, $\dim T$ is
`nb_degrees_of_freedom(proj_op)` $=n-m-p$.

```
Input : proj_op (S fixed), g, W, σ, buffers d̂, p,
        tol = tol_sign, tol_step = tol_noise, maxit = 10|W| + 10
Output: true if converged; proj_op encodes the working set 𝒲 of the next piece,
        d = −P_{T(𝒲)} g, θ the multipliers of g for 𝒲 (per variable)

set_active!(proj_op, W)                          # feasible start: every weak bound fixed
(d̂, θ) ← projected_direction!(proj_op, g)        # d̂ = −Pg, θ copied at once
d ← d̂ ;  stalled ← false
repeat maxit times
    p ← d̂ − d
    if ‖p‖∞ ≤ tol_step                           # d minimizes on the working set
        d ← d̂
        J ← {j ∈ W represented, σ_j ≠ 0 : σ_jθ_j < −tol}
        J == ∅ ⇒ return true                     # KKT conditions of §2.5
        j ← stalled ? min J : argmin_J σ_jθ_j    # Bland while stalled
        q ← dim T ;  set_free!(proj_op, j) ;  stalled ← true
        (d̂, θ) ← projected_direction!(proj_op, g)
        dim T == q ⇒ d̂ ← d                       # a redundant bound replaced j: same subspace
    else
        α ← 1 ;  b ← 0                           # ratio test, smallest index on exact ties
        for i ∈ W free with σ_i p_i < −tol_step
            r ← max(σ_i d_i, 0) / (−σ_i p_i)
            (r < α or (b > 0 and r == α and i < b)) ⇒ (α, b) ← (r, i)
        α > 0 ⇒ stalled ← false
        b == 0 ⇒ d ← d̂ ; continue               # full step, working set unchanged
        d ← d + αp
        q ← dim T ;  set_active!(proj_op, b)
        dim T ≠ q ⇒ (d̂, θ) ← projected_direction!(proj_op, g)  # b redundant: nothing changes
return false
```

- **Cost.** At most one projection per iteration, and none after a full step. At a
  nondegenerate breakpoint, a single hit changes nothing in the working set, and a single
  release costs one freeing and one full step.
- **Unchanged subspace.** Freeing $j$ can only bring into the factor a redundant bound $k$
  (`append_redundant!`). That bound was implied by the old working set, so
  $T_{\text{old}}\subseteq T_{\text{new}}$, and equal dimensions mean
  $T_{\text{new}}=T_{\text{old}}$: the minimizer is the same and only $\theta$ changes.
  - Without this shortcut, the re-projected $\hat d$ carried noise of about
    $10\,\epsilon\|g\|_\infty$. The freed bound blocked at $\alpha=0$ on that noise, was fixed
    again, and the loop cycled (doubly stochastic polytope, random $g$).
  - Likewise, a redundant blocking bound only enters `fixvars`, which leaves $P$, $\hat d$
    and $\theta$ unchanged.
- **Step tolerance.** The test $p\approx0$ and the blocking test use `tol_step`, the noise
  floor already used for vanishing directions. The returned $d$ may therefore point
  outwards along a free bound of $W$ by up to `tol_step`. The main loop parks these bounds.
- **Degeneracy.**
  - `stalled` means that the iterate has not moved since the last freeing. The leaving
    bound is then the smallest index with a wrong sign (Bland). Otherwise it is the most
    negative $\sigma_j\theta_j$.
  - The blocking bound is the smallest index among exact ties, which in practice are ties
    at $\alpha=0$.
  - Redundant bounds have no multiplier, so they never leave. A redundant blocking bound
    stays in `fixvars` as an implied constraint.
- **Stress test** (outside the test suite): 6,000 paths from the vertices of the doubly
  stochastic polytope (Gaussian and integer $g$), the octahedron and the segment (integer
  $g$, which produces many ties), with $\Delta\in\{0.05,0.3,2\}$.
  - The iteration cap was never reached, and a path had at most 7 pieces.
  - The path error against Dykstra was at most $1.1\cdot10^{-10}$, and the end of every
    path was LP-optimal to $1.2\cdot10^{-11}$.
  - This run came before the last two changes of §3.1, the ties measured on $s$ and the
    combinatorial fixing of the final bounds. Both only act on bounds within rounding
    errors of their value. The test suite, which covers the same polyhedra with both kinds
    of $g$ (§6.2, item 4), passes with the final code.
- **On failure** (`maxit` reached), $d$ is the current iterate. It is feasible for the cone
  problem and a descent direction: the iterates decrease $\tfrac12\|d+g\|^2$ from $d=0$, so
  $g^\top d\le-\tfrac12\|d\|^2$. $\theta$ then corresponds to $\hat d$, not to $d$.
- **Fallbacks**, not needed so far:
  - Goldfarb–Idnani dual method. It is designed for linearly dependent constraints, and
    the small-pivot test of `append_active!` is its dependence test.
  - Solving the polar problem as a nonnegative least-squares problem, as in the prototype:
    $d=-g-P_{K^\circ}(-g)$, with $K^\circ$ generated by $\pm$ rows of $A$, $-\sigma_ie_i$
    for $i\in W$, and $\pm e_i$ for $i\in S$.

---

## 4. Code changes

### 4.1 `src/cauchy.jl`
- **`cauchy_step!`**: split by projector type.
  - `CoordinateSubspaceProjector`: the former code, without the `eps_active` keyword.
  - `SubspaceProjector`: §3.1, with keywords `eta`, `theta`, `side`, `dhat`, `dstep`
    (buffers, allocated by default for the tests), `maxpieces` and `trace`.
  - The `eps_active` keyword is removed.
  - `trace`, e.g. a `Vector{Tuple{T,Vector{T},Vector{T}}}`, receives `(t, s, d)` at the
    start of each piece, then `(t_C, s_C, 0)`.
  - The loop condition `!found && !saturated_subspace(proj_op) && !isempty(idx)` is gone:
    a saturated subspace just gives $d=0$, which is handled as a stationary piece.
- **`initial_fixed!(::SubspaceProjector, …)`**: removed, replaced by
  `local_cone_projection!` (§3.2), called with `weak` = all bounds active at `x` (ε = 0).
  - The ε-active logic and the re-fix loop are dropped.
  - **The `zero_dir` trick is dropped** (fixing free variables with a zero component and
    freeing them at the end). Over a polyhedron, a zero component on one piece does not
    stay zero on later pieces. It is harmless only on boxes, and stays in
    `initial_fixed!(::CoordinateSubspaceProjector, …)`. Parking (§3.1) is a different
    mechanism: it only concerns free bounds that are at their value.
- **`next_breakpoint`**: unchanged.
- **New**: `next_release(η, θ, σ, proj_op; tol, epsbp)` → `(gap, idx)`.
- **New**: `local_cone_projection!(proj_op, g, d, θ, weak, σ, d̂, p; tol, tol_step, maxit)`
  → `converged::Bool`.
- **New**: `projected_direction!(d, θ, proj_op, g)`, i.e. $d\leftarrow-Pg$, followed by
  `copy_bound_multipliers!`.
- Docstrings are updated. The docstring of `cauchy_step!` claimed to follow
  $P(x-tg)-x$, which is now true; its definition of the path is also corrected.

### 4.2 `src/workspace.jl`
The inner workspace has the buffers `eta`, `theta` (`Vector{T}`, n), `side`
(`Vector{Int8}`, n), and `dhat`, `dstep` (`Vector{T}`, n) for the active-set loop.
`reset_workspace!` resets them. `cauchy_buffers(proj_op, workspace)` returns them as keyword
arguments of `cauchy_step!`, and returns an empty `NamedTuple` for a
`CoordinateSubspaceProjector`.

### 4.3 `src/solver.jl`
- `projected_gradient!`: no longer passes `eps_active` to `cauchy_step!`. It passes
  `cauchy_buffers(proj_op, workspace)...` instead.
- `solve_subproblem!` and `traulls`: keep `eps_active` (`eps_active_k = min(eps_active,
  reltol_crit)`) for `identify_active_set!` and `criticality_measure` only. That is the
  inner stopping test and the final test. The theory accepts any `eps_active > 0`, so the
  default `1e-3` stays, and the documentation no longer mentions a threshold.
- The docstrings of `eps_active` in `traulls`, `solve_subproblem!` and
  `projected_gradient!` (where the keyword is removed) are updated.

### 4.4 `src/polyhedral_constraints.jl`
The existing `set_active!`, `set_free!` (with `remove_active!` + `append_redundant!`),
`bound_multipliers`, `is_fixed`, `nb_degrees_of_freedom` and `fixpos` suffice. Two helpers
are added:
- `copy_bound_multipliers!(θ, P)`, which scatters `pbuf[1:p]` into an `n`-vector via
  `fixidx` and zeroes the other entries;
- `is_represented(P, i) = P.fixpos[i] > 0`.

### 4.5 Documentation
- `docs/src/method.md`, section "Cauchy step": rewritten for the exact path (hits and
  releases, local cone projection, degenerate vertices, stationary pieces, number of
  pieces). It now states that `eps_active` only affects the criticality measure.
- Section "Criticality measure": a paragraph on the ε-active set in the stopping tests is
  added.
- `docs/src/reference.md` lists `next_release`, `local_cone_projection!` and
  `projected_direction!`.

---

## 5. Numerical robustness checklist

All the items below are implemented; §3 gives the details.

- **Keep the active set combinatorial** (`fixvars` plus an explicit `side` vector). Do not
  re-detect activity from `s` with a tolerance at every piece. The Python prototype did
  that and produced spurious zero-length pieces (the path stayed exact).
  - The only test on `s` is the tie test at a breakpoint: it applies to free variables
    moving towards a bound, within `tol_tie` of it.
  - The bounds that are known to be at their value are fixed for CG combinatorially at the
    end. `update_inner_active_set!` only adds the bounds reached by coincidence.
- **Snap** `s_i` exactly onto its bound when it is hit. Before starting, set to 0 the
  `slow`/`supp` entries of the bounds active at `x`, with the same test as
  `identify_active_set!` (`near_lower_bound` / `near_upper_bound` with relative tolerance
  $\sqrt\epsilon$ and ε = 0).
- **Ties.** Treat as one event the event times within `epsbp`, and the hits within `tol_tie`
  on `s` (rounding errors of $s$ are $O(\epsilon\,t\|g\|_\infty)$, which `epsbp` does not
  follow). Then the local solve handles them.
- **Relative tolerances** (table at the top of §3):
  - `tol_dir` for $d\approx0$ and `tol_sign` for the signs of the multipliers in the cone
    projection, both relative to $\|g\|_\infty$;
  - `tol_noise` $=\sqrt\epsilon\,\|g\|_\infty$ for the slopes of the releases and the steps
    of the cone projection;
  - `tol_η` for strongly versus weakly active, relative to $\|tg+s\|_\infty$.
- **Rounding errors of the projection must not drive the combinatorics.** The cases below
  were all observed, except the zero-length hits, which parking prevents by construction.
  - An unchanged subspace keeps $\hat d$ exactly in the cone projection; otherwise it
    cycles.
  - Parked bounds get $d_j=0$; otherwise the path loops on zero-length hits.
  - Slopes below the noise floor do not release; otherwise releases appear at
    $t\approx10^{14}$.
- **Do not zero the fixed components of $d$**, so that $As=0$ holds to machine precision.
- **Safeguards.** `maxpieces` $=10(n-m)+10$: on overflow, stop at the current `s` (a
  feasible point with $\psi$ nonincreasing so far) and log a warning. The cone projection
  is capped at $10|W|+10$ iterations, with a warning at the end of the Cauchy step.

---

## 6. Tests (`test/cauchy_path.jl`, included from `runtests.jl`)

### 6.1 Reference projection without a new dependency
Use Dykstra's algorithm between $\{s\mid As=b\}$ (projector $I-A^+A$ via `pinv`, plus
$A^+b$) and the box $[s_l,s_u]$. The path tests use $b=0$. The random polyhedra of §6.2
use $b\neq0$ to compute $x=P_\Omega(z)$.

```julia
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
```

Against an exact QP solver on small degenerate and random polyhedra (n ≤ 12), the error was
≤ 1.1e-12, with a median of 150 iterations and a maximum of 1.4e4.

**Dykstra stalls far along the path.** At $t\gtrsim100$, it stopped up to $6\cdot10^{-3}$
from the projection with its stopping test satisfied. A stricter test, which also required
the corrections $p$ and $q$ to settle, did not help. In every such case, the computed path
agreed with an exact QP projection (JuMP + HiGHS) to $2\cdot10^{-12}$. The one exception,
at $1.3\cdot10^{-9}$, is presumably limited by the tolerances of the QP solver. So Dykstra
is not used beyond the end of the path.

**Path test.** For each piece recorded by `trace`:
- compare $s(t)$ with `proj_dykstra(A, slow, supp, -t*g)` at the start of the piece and
  at `nsample` points inside it (1 for the degenerate polyhedra, 0 for the random ones), and
  at the Cauchy point;
- relative tolerance 1e-8, relative to $\max(1,\|\text{ref}\|_\infty)$;
- `slow`, `supp` as returned by `cauchy_step!`, i.e. after snapping.

**End of the path.** With $H=0$, $\psi$ is nonincreasing along the whole path, so $s^C$ is
the end of the path, which minimizes $g^\top s$ over $D$ (it is the limit of
$P_D(-tg)$). The test checks that `pred` equals the value of that LP, solved with JuMP +
HiGHS (already dependencies of the package), to a relative tolerance of 1e-8.

**Step.** `check_step` also verifies $\|As\|_\infty\le10^{-10}$ and
$s_l-10^{-12}\le s\le s_u+10^{-12}$, and that every bound within $10^{-12}$ of its value
at $s^C$ is fixed in `proj_op`.

### 6.2 Cases and expected values (from the Python reference)

1. **Paper §3.1 example.** $A=[1\ 1\ 1]$, $l=(-1,-\infty,-\infty)$, $u=(\infty,1,\infty)$,
   $x=(-0.5,-2,2.5)$, $g=(0.3,-0.8,0.5)$.
   - $\Delta=100$: pieces $[0,5/3]$, $[5/3,55/13]$, $[55/13,10]$ stationary, $[10,985]$,
     then stationary for good. The directions are $(-0.3,0.8,-0.5)$, $(0,0.65,-0.65)$, $0$,
     $(0.1,0,-0.1)$, $0$.
   - With $H=0$: $s^C=(97,3,-100)$ and decrease 23.3.
   - With $H=\mathrm{diag}(1,0.5,2)$: $t^C=14/13\approx1.07692$,
     $s^C=(-0.323077,0.861538,-0.538462)$ and decrease 0.527692.
   - $\Delta=2$: pieces $[0,5/3]$, $[5/3,2.69231]$ (the TR bound on $s_2$ is hit),
     $[2.69231,5]$ stationary, $[5,10]$ along $(0.1,0,-0.1)$, then stationary. With $H=0$:
     $s^C=(0,2,-2)$ and decrease 2.6.
2. **Zigzag.** Same polyhedron and $g$, $x=(-1,1,0)+\delta(1,-3,2)$, $\Delta=1$, $H=0$.
   - Decrease at the Cauchy point: $0.2+3.3\delta$, i.e. 0.5300 / 0.2330 / 0.2033 /
     0.200033 for $\delta$ = 1e-1 / 1e-2 / 1e-3 / 1e-5. It must not tend to 0. For
     $\delta$ = 1e-3, $s^C=(0.997,0.003,-1)$.
   - Also check that stopping on the stationary piece would give $3.7\delta$. This guards
     the "skip stationary pieces" rule.
3. **Doubly stochastic polytope (stall of the former rule).** $A$, $x$ and $g$ as in §1,
   $\Delta=0.5$, $H=0$.
   - Expected: 2 pieces, breakpoint $t=2/9$, first direction
     $(-2.25,0,2.25,\ 2.25,0,-2.25,\ 0,0,0)$, $s^C=(-0.5,0,0.5,\ 0.5,0,-0.5,\ 0,0,0)$,
     decrease 4.5.
   - Checked: the former code returned `pred = 0` and $s=0$ here.
4. **Degenerate polyhedra, path versus reference.** For the segment $\{x_1+x_2=1\}\cap[0,1]^2$,
   the octahedron $\{\sum x=2\}\cap[0,1]^4$ and the doubly stochastic polytope:
   - points: the vertices, and $v+10^{-3}(v'-v)$ for the next vertex $v'$ in the list;
   - $g$ either $3\times$ standard normal or integer-valued in $\{-3,\dots,3\}$ (many
     ties), $\Delta\in\{0.05,0.3,2\}$;
   - checks: path error ≤ 1e-8, end of the path LP-optimal to 1e-8, at most 10 pieces,
     `check_step`.
   - Prototype: error ≤ 2e-13; pieces per path: median 2 / 3 / 3, max 2 / 6 / 7.
5. **Random polyhedra.** $A$ random normal $m\times n$, box $[-1,1]$, $x=P_\Omega(z)$ with
   $z\sim 3\,\mathcal N(0,I)$ (many active bounds).
   - $(n,m)=(40,8)$: three instances, $\Delta\in\{0.1,0.5,2\}$. $(100,20)$: one instance,
     $\Delta=0.5$, which keeps the test time reasonable.
   - checks: the path matches the reference at the start of every piece, the end of the
     path is LP-optimal, there are at most $2n$ pieces, `check_step`.
6. **First local minimizer.** Four cases: two on the polyhedron of the paper (close to the
   vertex, and the example point) and two on the doubly stochastic polytope (a vertex, and
   a point close to one). Each case uses $H=J^\top J$ with $J$ random of scale 0.1 or 1,
   and $\Delta\in\{0.3,2\}$.
   - `pred` equals $q(s^C)$.
   - Along the reference path sampled at 30 points, $\psi$ is nonincreasing on
     $[0,t^C]$, and $\psi(t^C)$ equals `pred`.
   - $t^C$ is a local minimizer: $\psi(t^C+h(1+t^C))\ge\psi(t^C)$ for
     $h\in\{10^{-6},10^{-4}\}$, which also covers the end of the path.
7. **Solve from the doubly stochastic vertex.** $\min\tfrac12\|x-x_0+g\|^2$ over the
   polytope, from $x_0$ and $g$ of item 3, with `eps_active` $\in\{10^{-3},0\}$. The solve
   ends `first_order_critical` at the projection of $x_0-g$. The former code ended
   `feasible_non_critical` at $x_0$ after 600 inner iterations.
8. **Regression.** All existing tests pass, including the box-only problems, which keep the
   `CoordinateSubspaceProjector` path.

### 6.3 Existing tests revisited (`test/polyhedral_solve.jl`)
"Vanishing projected direction in the Cauchy step", "Release of bounds with multipliers of
the wrong sign" and its sub-testsets, and "ε-active set strategy" tested the former
initial working-set rule and the ε-active Cauchy path. The problems and the assertions are
unchanged. The comments now describe the exact path: a stationary piece at the vertex,
then the release of $x_1\ge-1$, at the start of the path or after the stationary piece.
The last testset is renamed "ε-active set in the stopping test".

---

## 7. Validation on the benchmarks

Only five of the 82 instances keep an explicit linear constraint (HS14, HS22, HS23, HS32,
HS323), plus the 15 problems of §4.3 (Table 2). Rerun those, before and after, with the
`benchmark` scripts. Compare iterations, evaluations, final objective and failures, and
update §4.3, Table 2, and the paragraphs of §4 that quote these numbers. Box-only instances
take the unchanged path, so their figures should not move. Spot-check a few to confirm.

---

## 8. Paper changes (`sn-preprint-theorie.tex`)

- **Introduction, contributions (around line 225).** Rewrite the inner-solver item. The
  Cauchy point now follows the exact projected gradient path, computed with subspace
  projections only:
  - bounds are added and released as their multipliers cross zero;
  - a small cone projection is solved at $t=0$ and at degenerate breakpoints;
  - there is no QP over $\Omega$, as in Calamai–Moré, and no LP per trial step, as in Conn
    et al. 1993.

  Drop the narrative about the nested path being safeguarded by ε-active sets. The zigzag
  example can remain as motivation for following the exact path and for skipping
  stationary pieces.
- **§2.1.**
  - Remove Assumption `assumption:full_rank_active`. Keep `assumption:full_rank_A`.
  - The decomposition `eq:ls_decomposition` is unique only when the active rows are
    independent. State the minimum-sign-violation choice in general (it makes
    $\theta^{\mathrm w}$ unique).
  - The projector formula `eq:minor_subpb_projector` uses the represented rows.
  - Line ~292: ε_K no longer "allows the inner iterations to terminate". It only enters the
    criticality test of Conn et al. 1996.
- **§2.3 (lines ~471–476).** The Cauchy step is the first local minimizer along
  `eq:projected_gradient_path` over the polyhedron, not just over a box. Remove "the two
  paths may differ".
- **§2.5 `subsec:implementation_details`.**
  - Line ~689: remove "can only happen when it is reached together with other bounds or
    because of the trust region", and describe releases (rank-one downdate) during the
    Cauchy step.
  - Line ~719: remove "the convergence analysis requires $\varepsilon_{\max}\le\bar\varepsilon$"
    ($\bar\varepsilon$ disappears).
- **§3.1 `subsec:inner_convergence`.**
  - Replace the nested-path discussion. Keep the example as an illustration of a release
    and of a stationary piece.
  - Delete `lemma:eps_threshold`, `prop:computed_path`, `lemma:initial_direction`,
    `theo:cauchy_decrease` and the zigzag remark (or shorten the remark to "stopping on a
    stationary piece, or following nested faces, loses the decrease").
  - Insert the following results:
    - *Path properties* (standard; Calamai–Moré 1987 is already in the bib as
      `calamaimore:1987`). For convex $D\ni0$: $\langle g,s(t)\rangle\le-\|s(t)\|^2/t$,
      $\|s(t)\|$ is nondecreasing and $\|s(t)\|/t$ is nonincreasing. For polyhedral $D$:
      piecewise linear, finitely many pieces.
    - *Cauchy decrease.* $-q(s^C)\ge\kappa\,h(x)\min\big(h(x)/b_k,\ \Delta,\ 1\big)$ with
      $h(x)=\|P_\Omega(x-g)-x\|$, or with Conn–Gould–Toint's $\chi(x)$. Cite the TR book
      §12.2 and Burke, Moré & Toraldo, *Math. Prog.* 47 (1990). See open points 1–2 in §9.
    - *Theorem `theo:inner_convergence`.* Same proof with $h$ in place of $\pi_\varepsilon$.
      It gives $\liminf h(x_k)=0$, with no rank assumption.
    - *Link lemma (new).* If $p=P_\Omega(x-g)-x$ satisfies $\|p\|_\infty<\varepsilon$, then
      $\|P_{\mathcal T_\varepsilon(x)}(-g)\|=\operatorname{dist}(-g,\mathcal N_\varepsilon(x))\le\|p\|$.
      *Proof.* Every bound active at $x+p$ is ε-active at $x$, on the same side, so
      $\mathcal N_\Omega(x+p)\subseteq\mathcal N_\varepsilon(x)$. Moreover $-g-p\in\mathcal N_\Omega(x+p)$
      by the characterization of the projection. ∎
    - *Termination with $\pi_\varepsilon$, for any $\varepsilon>0$.* With minimum-violation
      multipliers,
      $$\pi_\varepsilon(x)\le\max(1,1/\sigma^+)\,\|P_{\mathcal T_\varepsilon(x)}(-g)\|,$$
      where $\sigma^+$ is the smallest *nonzero* singular value over the finitely many matrices
      $\tilde A_S=[A;E_S^\top]$, $S$ any set of bounds. **The constant does not depend on ε.**
      *Proof sketch.*
      - Write $-g=w^*+d$ with $d=P_{\mathcal T_\varepsilon}(-g)$ and
        $w^*=A^\top\xi'+E_S\tau\in\mathcal N_\varepsilon$.
      - Then $u=P_{T_\varepsilon}g=-P_{T_\varepsilon}d$, so $\|u\|\le\|d\|$.
      - Write the row-space part of $d$ as $\tilde A_S^\top y$ with the minimum-norm $y$, so
        $\|y\|\le\|d\|/\sigma^+$.
      - Then $\theta=-\tau-y_S$ represents $g$, and $-\tau$ has the right signs. So the
        minimum violation satisfies $\|\theta^{\mathrm w}\|_2\le\|y_S\|$. ∎

      Combining this with the link lemma: if $h(x)<\varepsilon$, then
      $\pi_\varepsilon(x)\le C\,h(x)$. Hence, if $\pi_\varepsilon(x_k)\ge\underline\pi$ for
      all $k$, then $h(x_k)\ge\min(\varepsilon,\underline\pi/C)>0$, which contradicts
      $\liminf h(x_k)=0$. So the stopping test is met after finitely many iterations.

      A variable ε-active at both bounds is fixed by the definition of
      $\mathcal T_\varepsilon$, so Lemma 1(iii) is not needed. Numerical check (random
      polyhedra, $n=12$, $m=4$, box $[-1,1]$, $\varepsilon\in\{0.01,0.3,1.5\}$, the last one
      with many variables ε-active at both bounds): the ratio
      $\pi_\varepsilon/(\max(1,1/\sigma^+)\,\mathrm{dist})$ never exceeded 0.96.
    - *What ε must satisfy.*
      - $\varepsilon_K>0$, for inner termination. With $\varepsilon=0$ the measure jumps: for
        $\Omega=\{x\ge0\}$, $g=1$ and $x_k=1/k$, $h(x_k)\to0$ but $\pi_0(x_k)=1$.
      - $\varepsilon_K\to0$, for Lemma `lemma:bound_proj_algrad_sol_nullspace` and the key
        lemma, whose proofs use nothing else about $\varepsilon_K$.

      $\varepsilon_K=\min(\varepsilon_{\max},\omega_K)$ with any $\varepsilon_{\max}>0$
      qualifies, and so does the final test with $\varepsilon_*=\min(\varepsilon_{\max},\omega_*)$.
      There is no threshold to satisfy.
- **§3.2.** The outer analysis is unchanged. Remove the sentence "Consistently with the
  well-posedness of the projector …, $\tilde A_*$ has full row rank", which is not used.
- **Lemma `lemma:certification`.** The right inequality holds for any multipliers. For the
  left inequality, replace $\sigma_{\min}$ by $\sigma^+$ with minimum-violation multipliers,
  or drop it. Update the paragraph after the lemma ("if $\varepsilon_{\max}\le\bar\varepsilon$
  …").
- **Appendix A (`appendix:cauchy_point_computation`, Algorithm `algo:cauchy_point`).**
  Rewrite following §2–§3 of this document:
  - events (hit/release), the $\eta$ update, the local cone projection, and skipping
    stationary pieces;
  - finiteness: each working set at most once;
  - remove the "at most $n-m$ breakpoints" claim and the $T(t_{i+1})=\{0\}$ stopping test.
- **§4.** Updated numbers for the instances with linear constraints (§7).
- **Bibliography.** Add Haraux 1977 (*J. Math. Soc. Japan* 29, "How to differentiate the
  projection on a convex set in Hilbert space. Some applications to variational
  inequalities"), Burke–Moré–Toraldo 1990, and possibly Goldfarb–Idnani 1983 and Mairal–Yu
  2012. `nocedalwright:2006`, `conn-etal:2000`, `calamaimore:1987` and `moretoraldo:1991`
  are already there.
- Update `CHANGEMENTS.md`.

---

## 9. Open points to check

1. **First local minimizer versus Goldstein-type conditions.** The convex-set results (TR
   book §12.2; Burke–Moré–Toraldo 1990) are stated for a point on the projected path that
   satisfies a sufficient-decrease condition and a "not too short" condition. I have not
   checked a proof that the *first local minimizer, skipping stationary pieces* always
   satisfies them over a polyhedron (for boxes, LANCELOT's exact generalized Cauchy point
   is covered by Conn, Gould & Toint 1988). Practical route: compute $t^C$, log whether the
   conditions hold, and if the theory needs it, backtrack or extend along the stored pieces.
   This costs no projection, since the path is piecewise linear and known.
2. **ℓ∞ trust region.** The path projects onto $\Omega\cap\{\|s\|_\infty\le\Delta\}$, as
   Conn, Gould & Toint do for boxes, rather than following $P_\Omega(x-tg)$ under a separate
   $\|s\|\le\Delta$ constraint. Check that the cited decrease result covers this variant, or
   derive it. Since $D\supseteq\min(1,\Delta)\cdot(\Omega-x)\cap B_\infty(0,1)$, the measures
   relative to $D$ and to $\Omega$ are comparable.
3. **Anti-cycling of the local active-set loop** on degenerate vertices.
   - Done in practice (§3.2). The first version cycled on the doubly stochastic polytope.
     The cause was rounding noise, not Bland's rule: a freed bound blocked on the noise of a
     re-projection whose subspace had not changed.
   - With the unchanged-subspace shortcut and the step tolerance, the stress test of §3.2
     never reached the iteration cap.
   - Bland's rule is not a proof of finiteness for this QP active-set method, so the cap and
     the warning stay.
4. **Worst-case number of pieces.** Report typical counts in the paper, and possibly the
   safeguard.
5. **Conditioning of $\pi_\varepsilon$ (practical, not a gap in the proof).** When the
   ε-active rows are nearly dependent, $\pi_\varepsilon$ can overestimate the distance to
   the ε-normal cone by up to $1/\sigma^+$. In the random tests one case had
   $\pi_\varepsilon=886$ for a distance of 1.6. This can delay inner termination but
   cannot prevent it, and the current code behaves the same way.

---

## Appendix A — Python reference implementation

Requires `numpy` and `scipy` only. The local direction uses a bounded least-squares problem
on the polar cone, and the release time comes from an LP, so it is robust to degeneracy but
slow. It is a reference for tests, not a template for the Julia code.

```python
import numpy as np
from scipy.optimize import linprog, lsq_linear

TOL = 1e-9

def active_sides(s, sl, su):
    lo = np.isfinite(sl) & (np.abs(s - sl) <= TOL * (1 + np.abs(sl)))
    up = np.isfinite(su) & (np.abs(s - su) <= TOL * (1 + np.abs(su)))
    return lo, up

def local_direction(A, g, s, t, sl, su):
    """Right derivative of t -> P_D(-t g): projection of -g onto the critical cone
    K = T_D(s) ∩ {d | <d, t g + s> = 0} (Haraux), computed as d = -g - P_{K°}(-g)."""
    n, m = len(g), A.shape[0]
    lo, up = active_sides(s, sl, su)
    c = t * g + s
    M = np.hstack([A.T, -np.eye(n)[:, lo], np.eye(n)[:, up], c[:, None]])
    k = M.shape[1]
    lb = np.r_[-np.inf * np.ones(m), np.zeros(k - m)]
    r = lsq_linear(M, -g, bounds=(lb, np.inf * np.ones(k)), tol=1e-15, method='bvls')
    d = -g - M @ r.x
    d[np.abs(d) < 1e-9 * (1 + np.linalg.norm(g))] = 0.0
    return d

def release_time(A, g, s0, t0, d, sl, su):
    """sup{t >= t0 : t g + s0 + (t-t0) d = A^T xi + sum eta_i e_i over the bounds that stay
    active along d, eta >= 0 at lower and <= 0 at upper bounds}  (an LP)."""
    n, m = len(g), A.shape[0]
    lo, up = active_sides(s0, sl, su)
    lo &= (d == 0); up &= (d == 0)
    E = np.hstack([np.eye(n)[:, lo], np.eye(n)[:, up]])
    k = E.shape[1]
    Aeq = np.hstack([-(g + d)[:, None], A.T, E])
    cost = np.zeros(1 + m + k); cost[0] = -1.0
    bounds = [(t0, None)] + [(None, None)] * m + [(0, None)] * lo.sum() + [(None, 0)] * up.sum()
    r = linprog(cost, A_eq=Aeq, b_eq=s0 - t0 * d, bounds=bounds, method="highs")
    if r.status == 3: return np.inf          # unbounded: never released
    if r.status == 2: return t0              # infeasible by rounding: zero-length piece
    return r.x[0]

def track(A, g, sl, su, tmax=np.inf, maxseg=10000):
    """Pieces (t_start, t_end, s_start, d) of s(t) = P_D(-t g)."""
    n = len(g); t, s = 0.0, np.zeros(n); segs = []
    while t < tmax and len(segs) < maxseg:
        d = local_direction(A, g, s, t, sl, su)
        thit = np.inf
        for i in np.nonzero(d)[0]:
            gap = (su[i] - s[i]) / d[i] if d[i] > 0 else (sl[i] - s[i]) / d[i]
            thit = min(thit, t + max(gap, 0.0))
        te = min(thit, release_time(A, g, s, t, d, sl, su), tmax)
        te = max(te, t + 1e-12)
        segs.append((t, te, s.copy(), d.copy()))
        if not np.isfinite(te): break
        s = np.clip(s + (te - t) * d, sl, su); t = te
    return segs

def first_local_min(segs, g, H):
    """First local minimizer of psi(t) = q(s(t)), skipping stationary pieces (d = 0)."""
    for (a, b, s0, d) in segs:
        if not np.any(d): continue
        slope, curv = g @ d + s0 @ H @ d, d @ H @ d
        if slope >= 0: return a, s0, g @ s0 + 0.5 * s0 @ H @ s0
        if curv > 0 and a - slope / curv < b:
            sc = s0 - slope / curv * d
            return a - slope / curv, sc, g @ sc + 0.5 * sc @ H @ sc
        if not np.isfinite(b): return np.inf, None, -np.inf   # unbounded model along a ray
    a, b, s0, d = segs[-1]
    sc = s0 + ((b - a) * d if np.isfinite(b) else 0)
    return b, sc, g @ sc + 0.5 * sc @ H @ sc
```

Usage: `sl = np.maximum(-Delta, l - x); su = np.minimum(Delta, u - x); segs = track(A, g, sl, su)`.
The test values of §6.2 were produced this way.

## Appendix B — Prototype results

| Case | Error vs exact projection | Pieces per path | Releases | Pieces to Cauchy point (convex $H$) |
|---|---|---|---|---|
| Paper §3.1 example | 1e-16 | 4 (Δ = ∞) | 1 | — |
| Segment $x_1+x_2=1$ | 3e-14 | median 2, max 2 | — | — |
| Octahedron | 2e-14 | median 3, max 6 | — | — |
| Doubly stochastic 3×3 | 2e-13 | median 3, max 7 | — | — |
| Random $n=40$, $m=8$ | 7e-12 | median 30, max 46 | median 4, max 10 | median 18, max 30 |
| Random $n=100$, $m=20$ | 6e-12 (one instance checked) | median 76, max 93 | median 11, max 22 | median 42, max 74 |

The degenerate cases were sampled at vertices and nearby points with random $g$ and
$\Delta\in\{0.05,0.3,2\}$; the random cases had box $[-1,1]$ and $\Delta\in\{0.1,0.5,2\}$.
Counts for the random cases use only pieces of positive length (see §5).
