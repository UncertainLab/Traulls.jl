# [Method](@id Method)

This page describes the algorithm implemented in `Traulls.jl`. A complete description, with
the convergence analysis, can be found in the preprint
[arXiv:2607.11239](https://arxiv.org/abs/2607.11239).

We consider the problem

```math
\begin{aligned}
\min_{x} \quad &  f(x) = \dfrac{1}{2} \|r(x)\|^2 \\
\text{s.t.} \quad & c(x) = 0\\
& Ax = b \\
& \ell \le x \le u,
\end{aligned}
```

and denote by $J(x) \in \mathbb{R}^{n_r \times n}$ and $C(x) \in \mathbb{R}^{n_c \times n}$
the Jacobian matrices of the residuals $r$ and of the constraints $c$. The $m \times n$ 
matrix $A$ is assumed to have full row rank with $m < n$.

!!! note "Inequality constraints"
    Nonlinear inequality constraints $g(x) \ge 0$ are converted into equality constraints
    $g(x) - v = 0$ by appending slack variables $v \ge 0$ to the vector of unknowns. The
    slack variables are initialized to $g(x_0)$. In the rest of this page, $x$ therefore
    includes the slack variables and $c$ gathers both the nonlinear equalities and the
    reformulated inequalities.

The method is made of three nested levels:

1. an [augmented Lagrangian outer loop](@ref method_al) (function [`traulls`](@ref));
2. a [trust-region inner loop](@ref method_subproblem) that approximately minimizes the
   augmented Lagrangian with respect to the nonlinear constraints, and subject to the linear constraints
   (function [`Traulls.solve_subproblem!`](@ref));
3. a [gradient projection method](@ref method_qp) that approximately solves the quadratic
   program defining each trust-region step
   (function [`Traulls.projected_gradient!`](@ref)).

The linear constraints $Ax = b$ and $\ell \le x \le u$ are never penalized and every iterate
remains feasible with respect to them. This presentation remains valid for problems where
the only linear constraints are the bounds $\ell \le x \le u$. 

Since the initial point is made 
[feasible with respect to the linear constraints](@ref method_init) and every step preserves
this feasibility, all the iterates satisfy $Ax_k = b$ and $\ell \le x_k \le u$.

## [Augmented Lagrangian outer loop](@id method_al)

For a penalty parameter $\mu > 0$ and a vector of Lagrange multipliers estimates
$y \in \mathbb{R}^{n_c}$, the augmented Lagrangian function is

```math
L_A(x, y, \mu) = \dfrac{1}{2}\|r(x)\|^2 + c(x)^T y + \dfrac{\mu}{2}\|c(x)\|^2,
```

and its gradient with respect to $x$ is

```math
\nabla_x L_A(x, y, \mu) = J(x)^T r(x) + C(x)^T \big(y + \mu c(x)\big).
```

Starting from an initial point $x_0$ and an initial estimate $y_0$ of the Lagrange
multipliers, each outer iteration $k$ computes $x_{k+1}$ as an approximate solution, with
respect to a criticality tolerance $\omega_k > 0$, of the subproblem

```math
\begin{aligned}
\min_{x} \quad & L_A(x, y_k, \mu_k) \\
\text{s.t.} \quad & Ax = b \\
& \ell \le x \le u,
\end{aligned}
```

using $x_k$ as a starting point, with fixed penalty parameter $\mu_k$ and multipliers $y_k$.

### Multipliers and penalty updates

The new iterate $x_{k+1}$ is then tested against a feasibility tolerance $\eta_k > 0$:

- If $\|c(x_{k+1})\|_\infty \le \eta_k$, the feasibility is considered as sufficiently
  improved. The Lagrange multipliers are updated by the first-order formula
  ```math
  y_{k+1} = y_k + \mu_k c(x_{k+1}),
  ```
  the penalty parameter is kept unchanged, $\mu_{k+1} = \mu_k$, and both tolerances are
  tightened:
  ```math
  \omega_{k+1} = \max\left(\dfrac{\omega_k}{\mu_k^{\beta_\omega}}, \omega_{\min}\right),
  \quad
  \eta_{k+1} = \max\left(\dfrac{\eta_k}{\mu_k^{\beta_\eta}}, \eta_{\min}\right).
  ```
- Otherwise, the multipliers are kept, $y_{k+1} = y_k$, and the penalty parameter is
  increased by a factor $\tau > 1$, $\mu_{k+1} = \tau \mu_k$. The tolerances are
  reduced in a weaker manner:
  ```math
  \omega_{k+1} = \max\left(\dfrac{\omega_0}{\mu_{k+1}^{\kappa_\omega}}, \omega_{\min}\right),
  \quad
  \eta_{k+1} = \max\left(\dfrac{\eta_0}{\mu_{k+1}^{\kappa_\eta}}, \eta_{\min}\right).
  ```

The initial tolerances are $\omega_0/\mu_0^{\kappa_\omega}$ and $\eta_0/\mu_0^{\kappa_\eta}$.
By default, the initial multipliers $y_0$ are the least-squares estimates, i.e. the solution
of $\min_y \|J(x_0)^T r(x_0) + C(x_0)^T y\|$ (keyword `init_mult`); otherwise $y_0 = 0$.

The constants above correspond to the following keyword arguments of [`traulls`](@ref):

| Symbol | Keyword | Default |
|:------ |:------- |:------- |
| $\mu_0$ | `mu` | `10` |
| $\tau$ | `tau` | `10` |
| $\omega_0$, $\eta_0$ | `omega0`, `eta0` | `1`, `1` |
| $\kappa_\omega$, $\kappa_\eta$ | `k_crit`, `k_feas` | `1`, `0.1` |
| $\beta_\omega$, $\beta_\eta$ | `beta_crit`, `beta_feas` | `1`, `0.9` |
| $\omega_{\min}$ | `min_reltol_crit` | `1e-7` |
| $\eta_{\min}$ | `min_tol_feas` | `1e-7` |

## [Solving the subproblem of an outer iteration](@id method_subproblem)

The subproblem of an outer iteration is solved by a trust-region method in which the
multipliers $y$ and the penalty parameter $\mu$ are fixed. To simplify notations, we write
$L_A(x)$ for $L_A(x, y, \mu)$ in this section.

At inner iteration $j$, a quadratic model of the augmented Lagrangian around the current
iterate $x_j$ is formed:

```math
q_j(s) = \dfrac{1}{2} s^T H_j s + s^T g_j,
```

with $g_j = \nabla_x L_A(x_j)$ and $H_j$ an approximation of $\nabla^2_{xx} L_A(x_j)$ (see
[Hessian approximation](@ref method_hessian)). The trial step $s_j$ is an approximate
solution of the quadratic program

```math
\begin{aligned}
\min_{s} \quad & q_j(s) \\
\text{s.t.} \quad & As = 0 \\
& \ell \le x_j + s \le u \\
& \|s\|_\infty \le \Delta_j,
\end{aligned}
```

where $\Delta_j > 0$ is the trust-region radius. The trust region is defined with the
$\infty$-norm, so that it combines with the bounds into a single box: the constraints on
the step are equivalent to $s \in B_j$ with

```math
B_j = \big[\max(-\Delta_j e, \ell - x_j),\ \min(\Delta_j e, u - x_j)\big],
\quad e = (1, \dots, 1)^T.
```

- In the **bound-constrained case**, the feasible set of the QP is the box $B_j$ only.
- In the **general linear constraints case**, the step must also lie in the null space of
  $A$, i.e. the feasible set is $\{s \in B_j \mid As = 0\}$. Since $Ax_j = b$, this
  guarantees that $A(x_j + s_j) = b$.

The way this QP is solved is detailed in the
[next section](@ref method_qp).

### Step acceptance and radius update

The trial step is accepted or rejected depending on the ratio of the actual reduction of the
augmented Lagrangian over the reduction predicted by the model

```math
\rho_j = \dfrac{L_A(x_j + s_j) - L_A(x_j)}{q_j(s_j) - q_j(0)}.
```

If $\rho_j \ge \eta_1$, the model and the function are in good agreement: the step is
accepted, $x_{j+1} = x_j + s_j$, and the Jacobians, the gradient and the Hessian
approximation are updated. Otherwise, the step is rejected, $x_{j+1} = x_j$, and a new
step is computed in a smaller trust region. The radius is updated as follows, where
$\|s_j\|$ denotes the $\infty$-norm of the step:

| Ratio | Step | New radius $\Delta_{j+1}$ |
|:----- |:---- |:------------------------- |
| $\rho_j > \eta_2$ | very successful | $\max(\alpha_2 \|s_j\|, \Delta_j)$ |
| $\eta_1 \le \rho_j \le \eta_2$ | successful | $\Delta_j$ |
| $0 < \rho_j < \eta_1$ | unsuccessful | $\alpha_1 \|s_j\|$ |
| $\rho_j < 0$ | very unsuccessful | $\min(\alpha_1 \|s_j\|, \gamma \Delta_j)$ |

The constants satisfy $0 < \eta_1 \le \eta_2 < 1$, $0 < \alpha_1 < 1 < \alpha_2$ and
$0 < \gamma < 1$. They correspond to the keyword arguments `accept_threshold`
($\eta_1 = 0.05$), `increase_threshold` ($\eta_2 = 0.9$), `decrease_factor`
($\alpha_1 = 0.25$), `increase_factor` ($\alpha_2 = 2.5$) and `neg_ratio_factor`
($\gamma = 0.0625$) of [`traulls`](@ref).

At the start of each outer iteration, the radius is initialized to
$\Delta_0 = 0.1\|g_0\|_\infty$ and the Hessian approximation is reset.

## [Solving the quadratic program](@id method_qp)

The QP defining the step is approximately solved by a gradient projection method in two
phases: a Cauchy step that guarantees a sufficient decrease of the model, followed by
projected conjugate gradient (CG) iterations that improve this decrease.

Both phases rely on a projection operator $P$ onto the subspace of the directions that
preserve the constraints currently fixed as active. Let $\mathcal{A}$ denote the set of
indices of the components fixed at one of their bounds.

- In the **bound-constrained case**, $P$ is the projection onto the coordinate subspace
  $\{v \mid v_i = 0,\ i \in \mathcal{A}\}$: it merely sets to zero the components of a
  vector indexed by $\mathcal{A}$.
- In the **general linear constraints case**, $P$ is the orthogonal projection onto the
  null space of the matrix $A_+$ formed by stacking $A$ and the rows $e_i^T$,
  $i \in \mathcal{A}$, of the identity matrix:
  ```math
  P v = v - A_+^T \big(A_+ A_+^T\big)^{-1} A_+ v.
  ```
  The Cholesky factorization of $AA^T$ is computed once at the start of the algorithm. The
  one of $A_+ A_+^T$ is derived from it by exploiting its block structure each time the
  set $\mathcal{A}$ changes.

### Cauchy step

The Cauchy step is the first local minimizer of the model $q_j$ along the projected
gradient path.
  ```math
  s(t) = \mathcal{P}\big[x_j - t g_j\big] - x_j, \quad t \ge 0,
  ```
where $\mathcal{P}$ is the projection onto the feasible set. This path is piecewise 
linear: its breakpoints are the values of $t$ at which a component reaches a bound of 
$B_j$. The model restricted to each segment is a one-dimensional quadratic, so the 
segments are examined successively until the first local minimizer is found.

The path is built as a sequence of segments along the directions $d = -P g_j$ where $P$
is the projection onto $\{s \mid As = 0, \ s_i = 0 \ i \in \mathcal{A}\}$: the search 
moves along $d$ until a component reaches a bound of $B_j$ (breakpoint), this bound is 
added to $\mathcal{A}$, the projector $P$ is updated and a new direction is computed. 

Before computing the path, the components that lie at a bound with a direction pointing
outwards the feasible set are fixed in $\mathcal{A}$. The search stops when a local
minimizer is found on a segment, or when no free direction remains.

The Cauchy step provides a sufficient reduction of the model, which is enough to guarantee
the convergence of the trust-region method.

### Beyond the Cauchy point

To obtain a better reduction, the step is improved by applying the conjugate gradient
method to the QP restricted to the subspace defined by the set $\mathcal{A}$ of bounds
active at the Cauchy point. Starting from the current step $s$, CG approximately solves

```math
\begin{aligned}
\min_{w} \ & \dfrac{1}{2} w^T H_j w + w^T (H_j s + g_j) \\
\text{s.t.} \quad & Aw = 0 \\
& w_i = 0,\ i \in \mathcal{A}
\end{aligned}
```

using $P$ as a preconditioner, so that all the search directions remain in the subspace.
CG iterations stop when:

- the norm of the projected residual $\|Pr\|$ falls below $\omega(1 + \|Pr_0\|)$, where
  $r_0$ is the initial residual and $\omega$ the current criticality tolerance of the
  outer loop;
- a direction crosses the boundary of the box $B_j$: the step is then truncated at the
  boundary;
- a direction of nonpositive curvature is found, which can happen with the SR1 updates:
  the step then follows this direction up to the boundary of $B_j$;
- the number of iterations exceeds twice the number of degrees of freedom of the subspace.

After each CG run, the components of the step that reached a bound of $B_j$ are added to
$\mathcal{A}$ and a new CG run is performed on the reduced subspace. This process is
repeated until the step satisfies the criterion given in the
[stopping criteria](@ref method_stop) section, a negative curvature is detected, no degree
of freedom remains, or `max_cg_iter` CG runs have been performed.

- In the **bound-constrained case**, the projections are cheap and the number of degrees of
  freedom is the number of free variables $n - |\mathcal{A}|$.
- In the **general linear constraints case**, each projection requires solving a linear
  system with the Cholesky factors of $A_+ A_+^T$, and the number of degrees of freedom is
  $n - m - |\mathcal{A}|$.

## [Hessian approximation](@id method_hessian)

### Structure of the Hessian

The Hessian of the augmented Lagrangian writes

```math
\nabla^2_{xx} L_A(x, y, \mu) =
J(x)^T J(x) + \mu C(x)^T C(x)
+ \underbrace{\sum_{i=1}^{n_r} r_i(x) \nabla^2 r_i(x)
+ \sum_{i=1}^{n_c} \big(y_i + \mu c_i(x)\big) \nabla^2 c_i(x)}_{S(x)}.
```

The first-order terms $J^T J + \mu C^T C$ only involve the Jacobians, which are already
computed to form the gradient. Only the second-order terms $S(x)$, which require the
second derivatives of the residuals and of the constraints, are approximated. The Hessian
approximation therefore has the form

```math
H_j = J_j^T J_j + \mu C_j^T C_j + S_j,
```

where $J_j = J(x_j)$, $C_j = C(x_j)$ and $S_j \approx S(x_j)$. The matrix $H_j$ is never
formed explicitly: only Hessian-vector products are computed, which avoids matrix-matrix
products with the Jacobians.

### Structured secant equation

After an accepted step $s_j = x_{j+1} - x_j$, the approximation $S_{j+1}$ is required to
satisfy the structured secant equation

```math
S_{j+1} s_j = \hat{y}_j,
\quad \text{with} \quad
\hat{y}_j = \big(J_{j+1} - J_j\big)^T r(x_{j+1})
+ \big(C_{j+1} - C_j\big)^T \big(y + \mu c(x_{j+1})\big).
```

The right-hand side $\hat{y}_j$ is obtained by subtracting the first-order terms from the
difference of the gradients of the augmented Lagrangian. It only involves quantities that
are already available, i.e. the residuals, constraints and Jacobians at $x_j$ and
$x_{j+1}$. Since $y$ and $\mu$ change from one outer iteration to the next, the second-order
approximation is reset at the start of each outer iteration.

### Hybrid switching

When the residuals and the terms $y + \mu c$ are small
at the solution, the second-order terms $S(x)$ are negligible and the Gauss-Newton
approximation $J^T J + \mu C^T C$ performs well. On the contrary, when they are large,
neglecting $S(x)$ may significantly slow down the convergence.

The hybrid schemes select between the two approximations at every accepted step, based on
the relative decrease of the augmented Lagrangian:

```math
L_A(x_j) - L_A(x_{j+1}) > \varepsilon \, L_A(x_j), \quad \varepsilon = 0.1.
```

If this inequality holds, the objective decreases fast, which indicates a small residual
problem: the Gauss-Newton model is used for the next iteration, i.e. $S_{j+1}$ is dropped
from the Hessian-vector products. Otherwise, the structured approximation
$J^T J + \mu C^T C + S_{j+1}$ is used. In both cases, $S_{j+1}$ keeps being updated with the
secant pairs $(s_j, \hat{y}_j)$, so that it is available when the switching test fails.

### Available approximations

The Hessian approximation is selected with the keyword argument `hessian_approx` of
[`traulls`](@ref):

| Keyword | Approximation |
|:------- |:------------- |
| `:gn` (default) | Gauss-Newton: $S_j = 0$ |
| `:sr1` | $S_j$ updated by the structured SR1 formula |
| `:bfgs` | $S_j$ updated by the structured BFGS formula |
| `:hybrid_sr1` | Hybrid switching between Gauss-Newton and structured SR1 |
| `:hybrid_bfgs` | Hybrid switching between Gauss-Newton and structured BFGS |
| `:limited_sr1` | $S_j$ approximated by a limited memory SR1 matrix |

For the SR1 based approximations, $S$ is initialized to zero at each outer iteration and an
update is skipped when its denominator is too small. For the BFGS based approximations, $S$
is initialized to the identity matrix, rescaled at the first update, and an update is
skipped when the curvature condition $s_j^T \hat{y}_j > 0$ does not hold with a sufficient
margin. Contrary to BFGS, SR1 updates may produce indefinite approximations, which are
handled by the negative curvature detection of the CG iterations.

## [Stopping criteria](@id method_stop)

### Criticality measure

The criticality of a point $x$ with respect to a gradient vector $g$ is measured by a
quantity $\pi(x, g)$ that depends on the structure of the linear constraints.

- In the **bound-constrained case**, it is the $\infty$-norm of the projected gradient step
  ```math
  \pi(x, g) = \big\|P_{[\ell, u]}[x - g] - x\big\|_\infty,
  ```
  where $P_{[\ell, u]}$ is the projection onto the box $[\ell, u]$.
- In the **general linear constraints case**, it is the $\infty$-norm of the reduced
  gradient
  ```math
  \pi(x, g) = \|P g\|_\infty,
  ```
  where $P$ is the projection onto the null space of the matrix $A_+$ formed by $A$ and the
  bounds active at $x$ (see [Solving the quadratic program](@ref method_qp)).

### Outer loop

The algorithm terminates successfully when the current point $x_k$ satisfies both

```math
\|c(x_k)\|_\infty \le \eta_{\min}
\quad \text{and} \quad
\pi\big(x_k, J(x_k)^T r(x_k) + C(x_k)^T y_k\big) \le \epsilon_{\text{crit}},
```

i.e. it is feasible and first-order critical for the gradient of the Lagrangian. The
tolerance $\epsilon_{\text{crit}}$ is $\epsilon_{\text{crit}} = \omega_{\min}(1 + \pi_0)$,
  where $\pi_0$ is the criticality measure at the initial point;

The algorithm also stops when the number of outer iterations exceeds `max_iter` or when the
penalty parameter reaches `mu_max`. The status returned in [`Traulls.TraullsResults`](@ref)
is one of:

- `first_order_critical`: both the feasibility and the criticality tests are satisfied;
- `feasible_non_critical`: the point is feasible, but the criticality test failed;
- `penalty_too_high`: the penalty parameter reached its maximum value before feasibility is
  achieved;
- `infeasible_non_critical`: the maximum number of outer iterations is reached without
  feasibility.

### Inner loop

The inner minimization of an outer iteration stops at the first iterate $x_j$ such that

```math
\pi\big(x_j, \nabla_x L_A(x_j, y_k, \mu_k)\big) \le \omega_k \big(1 + \pi_0^{(k)}\big),
```

where $\pi_0^{(k)}$ is the criticality measure at the starting point $x_k$ of the inner
loop. The inner loop is also stopped early when

- three consecutive accepted steps provide a relatively small change in both the iterate
  ($|s_i| \le 10^{-7}(1 + |x_i|)$ for all $i$) and the augmented Lagrangian
  ($|L_A(x_{j+1}) - L_A(x_j)| \le 10^{-10} \max(1, |L_A(x_j)|)$);
- the trust-region radius becomes too small to make relevant progress:
  $\Delta_j \le 10\,\epsilon_M (1 + \|x_j\|_\infty)$, where $\epsilon_M$ is the relative
  machine precision;
- the number of inner iterations exceeds `max_inner_iter`.

### Quadratic program

After the Cauchy step and after each CG run, the step $s$ is considered as a good enough
approximate solution of the QP when the gradient of the model at $s$, projected onto the
subspace of the currently active constraints, is small relatively to the projected
gradient:

```math
\big\|P (H_j s + g_j)\big\| \le \omega_k \big(1 + \|P g_j\|\big).
```

The projection $P$ is the one described in the section
[Solving the quadratic program](@ref method_qp), i.e. a coordinate projection in the
bound-constrained case and a projection onto the null space of $A_+$ in the general linear
constraints case.

## [Initial point](@id method_init)

The algorithm requires an initial point feasible with respect to the linear constraints.
The initial guess $x_0$ provided by the user is modified accordingly before the first outer
iteration.

- In the **bound-constrained case**, $x_0$ is simply projected onto the box:
  ```math
  x_0 \leftarrow \max\big(\ell, \min(x_0, u)\big).
  ```
- In the **general linear constraints case**, if $\|Ax_0 - b\| > \sqrt{\epsilon_M}$, the
  initial point is replaced by the solution of the $\ell_1$ feasibility problem
  ```math
  \begin{aligned}
  \min_{x, v} \quad & \|v\|_1 \\
  \text{s.t.} \quad & Ax + v = b \\
  & \ell \le x \le u,
  \end{aligned}
  ```
  which is a linear program solved with [HiGHS](https://highs.dev) through
  [JuMP](https://jump.dev), using $x_0$ as a starting point. A warning is emitted if the
  optimal value is nonzero, i.e. if no point satisfies the linear constraints. The bounds
  active at the resulting point then define the initial projection operator.
