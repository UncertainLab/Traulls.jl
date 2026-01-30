export tralcnllss

const verbose = false
const output_file_name = "../test/benlsip.out"


mutable struct GN{T<:Real} <: ALHessian
    J::Matrix{T}
    C::Matrix{T}
    mu::T 
end

"""
    CG_status

Enum representing the termination status of the projected conjugate gradient method:

- `solved`: The subproblem was solved successfully.
- `bound_hit`: The search direction hit a bound constraint.
- `negative_curvature`: Negative curvature was detected.
- `max_iter_reached`: The maximum number of iterations was reached.
"""
@enum CG_status solved bound_hit negative_curvature max_iter_reached


function second_derivatives(
    Jx::Matrix{T},
    Cx::Matrix{T},
    mu::T) where T 

    return GN(Jx,Cx,mu)
end

""" vthv(H,v)

The quadratic term `vᵀHv` where `H = JᵀJ + μCᵀC` is the Gauss-Newton approximation of the augmented Lagrangian Hessian encoded into
the [`GN`](@ref) type.
"""
function vthv(H::GN{T}, v::Vector{T}) where T
    Jv = H.J*v
    Cv = H.C*v 
    return dot(Jv,Jv) + H.mu*dot(Cv,Cv)
end 

""" Base.:*(H::GN, v::Vector)

Overload the `*` operator to be compatible with the type [`GN`](@ref).
"""
function Base.:*(H::GN{T}, v::Vector{T}) where T
    Jv = H.J*v 
    muCv = H.mu*H.C*v 
    return H.J' * Jv + H.C' * muCv
end



""" s_inner_hs(s,mu,J,C)

Evaluate at `s` the quadratic term `sᵀHs` where `H = JᵀJ + μCᵀC` is the Gauss-Newton approximation of the augmented Lagrangian Hessian.
"""
function s_inner_hs(
    s::Vector{T},
    mu::T,
    J::Matrix{T}, 
    C::Matrix{T}) where T

    Js = J*s
    Cs = C*s 
    return dot(Js,Js) + mu*dot(Cs,Cs)
end

""" hs(s,mu,J,C)

Evaluate at `s` the matrix-vector product `Hs` where `H = JᵀJ + μCᵀC` is the Gauss-Newton approximation of the augmented Lagrangian Hessian.
"""
function hs(
    s::Vector,
    mu::T,
    J::Matrix{T},
    C::Matrix{T}) where T

    Js, muCs = J*s, mu*C*s 
    return J'*Js + C'*muCs
end
#### The solver 

### Constraints relatite methods

function is_feasible(
    x::Vector{T}, 
    A::Matrix{T}, 
    x_l::Vector{T}, 
    x_u::Vector{T};
    b::Vector{T}=zeros(T,size(A,2))) where T

    return isapprox(A*x,b) && all(x_l .<= x) && all(x .<= x_u)
end

### Tolerances relative methods
function initial_tolerances(
    mu::T,
    omega0::T,
    eta0::T,
    k_crit::T,
    k_feas::T) where T

    omega = omega0 / (mu^k_crit)
    eta = eta0 / (mu^k_feas)
    return omega, eta
end

#= TRALCNLSS stands for Trust Region Augmented Lagrangian Constrainted Nonlinear Least Squares Solver =#

function tralcnllss(
    x0::Vector{T},
    residuals::F1,
    jac_res::F2,
    nlconstraints::F3,
    jac_nlcons::F4,
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T};
    mu0::T = T(10),
    tau::T = T(100),
    omega0::T = T(1),
    eta0::T = T(1),
    feas_tol::T = sqrt(eps(T)),
    crit_tol::T = sqrt(eps(T)),
    k_crit::T = T(1),
    k_feas::T = T(0.1),
    beta_crit::T = T(1),
    beta_feas::T = T(0.9),
    eta1::T = T(0.25),
    eta2::T = T(0.75),
    gamma1::T = T(0.0625),
    gamma2::T = T(2),
    gamma_c::T = T(10),
    kappa1::T = T(1e-2),
    kappa2::T = T(0.1),
    kappa3::T = T(0.1),
    max_outer_iter::Int = 500,
    max_inner_iter::Int = 500,
    max_minor_iter::Int = 50) where {T<:Real, F1<:Function, F2<:Function, F3<:Function, F4<:Function}

    # Sanity check
    @assert (0 < eta1 <= eta2 < 1) && (0 < gamma1 < 1 < gamma2) "Invalid trust region updates paramaters"
    @assert is_feasible(x0,A,x_l,x_u;b=b)

    output_file = open(output_file_name, "w")
    
    # Initializations
    (m,n) = size(A)
    chol_aat = cholesky(A*A') # Cholesky decomposition of AAᵀ
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    rx = residuals(x)
    cx = nlconstraints(x)
    mu = mu0

    verbose && print_tralcnllss_header(n,
    size(rx,1),
    size(cx,1),
    m,
    x_l,
    x_u,
    feas_tol,
    crit_tol,
    tau,
    eta1,
    eta2,
    gamma1,
    gamma2;
    io=output_file)


    omega, eta = initial_tolerances(mu0, omega0, eta0, k_crit, k_feas) # tolerances 
    y = least_squares_multipliers(x,residuals,jac_res,jac_nlcons) # Initial Lagrange multipliers 
    polyhedron = MixedConstraints(A,chol_aat;l=x_l,u=x_u) # representation of pohydral constraints
    # println("initial multipiers ", y)

    first_order_critical = false
    outer_iter = 1

    verbose && print_outer_iter_header(
        outer_iter,
        dot(rx,rx),
        norm(cx),
        mu,
        0.0,
        omega;
        io = output_file,
        first=true)
    while !first_order_critical && outer_iter <= max_outer_iter


        x_next, cx_next, pix = solve_subproblem(
            x,
            y,
            mu,
            residuals,
            nlconstraints,
            jac_res,
            jac_nlcons,
            chol_aat,
            polyhedron,
            max_minor_iter,
            max_inner_iter,
            omega,
            eta1,
            eta2,
            gamma1,
            gamma2,
            kappa2,
            kappa3;
            output_file=output_file)


        feas_measure = norm(cx_next)

        if feas_measure <= eta
            x .= x_next
            cx .= cx_next
            first_order_critical = pix <= crit_tol && feas_measure <= feas_tol

            if !first_order_critical
                # Update the iterate, multipliers and decrease tolerances (penalty parameter is unchanged)
                y = first_order_multipliers(y,cx,mu)
                omega /= mu^(beta_crit)
                eta /= mu^(beta_feas)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances,  (iterate and multipliers are unchanged)
            mu *= tau
            omega = omega0 / (mu^k_crit)
            eta = eta0 / (mu^k_feas)   
        end

        outer_iter += 1
        objective = begin rx = residuals(x); dot(rx,rx) end
        verbose && print_outer_iter_header(outer_iter, objective, feas_measure, mu, pix, omega;io=output_file)

    end
    close(output_file)
    return x, y
end


#= Solves the sub problem of an outer iteration
Approximately minimize the Augmented Lagrangian function with respect to the primal variable with tolerance ω > 0=#
function solve_subproblem(
    x0::Vector{T},
    y::Vector{T},
    mu::T,
    residuals::F1,
    nlconstraints::F2,
    jac_res::F3,
    jac_nlcons::F4,
    chol_aat::Cholesky{T,Matrix{T}},
    lincons::MixedConstraints{T},
    nb_minor_step::Int,
    k_max::Int,
    omega_tol::T,
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T,
    kappa2::T,
    kappa3::T;
    output_file::IO=stdout) where {T<:Real, F1<:Function, F2<:Function, F3<:Function, F4<:Function}

    # Dimensions

    (_,n) = size(lincons.lineq)
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    rx, cx, y_bar, mx, g, H = new_point(x0,y,mu,residuals,nlconstraints,jac_res,jac_nlcons) 

    pix = Inf
    delta = initial_tr(g)
    k = 1
    solved = false
    
    while !solved && k <= k_max
        # step and model reduction
        s, pred = inner_step(x,
        g,
        H,
        chol_aat,
        lincons,
        delta,
        nb_minor_step,
        kappa2,
        kappa3)

        x_next = x+s
        rx_next, cx_next, mx_next = evaluate_al(x_next,y,mu,residuals,nlconstraints)
        ared = mx_next - mx
        rho = ared / pred

        verbose && print_inner_iter(k,mx,norm(s),delta,rho;io=output_file)

        if rho > eta1
            x .= x_next
            rx[:], cx[:], mx = rx_next[:], cx_next[:], mx_next
            y_bar, J, C, g = first_derivatives(x,y,mu,rx,cx,jac_res,jac_nlcons)
            H = second_derivatives(J,C,mu)
        end

        # Update trust region radius
        delta = update_tr(delta, rho, eta1, eta2, gamma1, gamma2)

        # Compute criticality measure
        pix = criticality_measure(g,lincons)
        #pix = criticality_measure(x,g,A,b,x_l,x_u)
        
        # Termination criteria 
        solved = pix < omega_tol
        k += 1
    end

    return x, cx, pix
end

""" inner_step(x,g,H,A,chol_AAᵀ,b,xₗ,xᵤ,Δ,nb_minor_step,κ₀,κ₂,κ₃,γc)

Starting from the current iterate `x`, compute a step `s` such that the inner step `x+s` sufficiently reduces the model.

On return 

* the step `s`

* the scalar `t_c` used to compute the Cauchy point 

* the reduction predicted by the model with step `s`

* indices of acitve bounds encoded in a `BitVector` and the associated Cholesky decomposition
"""
function inner_step(
    x::Vector{T},
    g::Vector{T},
    H::GN{T},
    chol_aat::Cholesky{T,Matrix{T}},
    lincons::MixedConstraints{T},
    delta::T,
    nb_minor_step::Int,
    kappa2::T,
    kappa3::T,
    ) where T

    (m,n) = size(lincons.lineq)

    # s, t_c = cauchy_step(x,g,H,A,b,x_l,x_u,delta,t0,kappa1,gamma_c)
    # s, fix_bounds = cauchy_step(x,g,H,A,chol_aat,x_l,x_u,delta)
    s = cauchy_step(x,g,H,chol_aat,lincons,delta)

    g_minor = H*s+g
    # chol_aug_aat = cholesky_aug_aat(A,fix_bounds,chol_aat)
    # fix_bounds, chol_aug_aat = active_w_chol(s,x,x_l,x_u,delta,A,chol_aat)

    j = 1 # minor iterations index

    # norm_reduced_g = norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
    # norm_reduced_g_minor =  norm_reduced_gradient(g_minor,A,fix_bounds,chol_aug_aat)
    norm_reduced_g = norm_reduced_gradient(g,lincons)
    norm_reduced_g_minor = norm_reduced_gradient(g_minor,lincons)

    approx_solved = norm_reduced_g_minor <= kappa3 * norm_reduced_g

    allowed_minor_step = max(n-m-nb_fix(lincons))
    max_minor_step = min(nb_minor_step, allowed_minor_step)
    cg_stop  = false

    # Minor iterates loop
    while j <= max_minor_step && !approx_solved && !cg_stop
        # println("[inner_step] minor iterate ", j)

        # descent direction and termination status of the cg iterations
        w, cg_status = minor_iterate(x,s,g_minor,H,lincons,delta,kappa2)
        cg_stop = cg_status == negative_curvature
        s .+= w # cumulated step 
        g_minor = H*s+g
        # fix_bounds, chol_aug_aat = active_w_chol(s,x,x_l,x_u,delta,A,chol_aat) # New active set
        active_indx = active_bounds(lincons,x,s,delta)

        if m+size(active_indx,1) <= n
            add_active!(lincons,chol_aat,active_indx)
            # Loop termination criteria
            # norm_reduced_g = norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
            # norm_reduced_g_minor =  norm_reduced_gradient(g_minor,A,fix_bounds,chol_aug_aat)
            norm_reduced_g = norm_reduced_gradient(g,lincons)
            norm_reduced_g_minor = norm_reduced_gradient(g_minor,lincons)
            approx_solved = norm_reduced_g_minor <= kappa3 * norm_reduced_g

        else # small step in a small trust region
            approx_solved = true
            active_bounds!(lincons,x+s,chol_aat) # set the right active set 
        end
        j += 1
    end

    # Evaluate the reduction of the quadratic model 
    model_reduction = dot(g,s) + 0.5*vthv(H,s)
    return s, model_reduction
end

function cauchy_step(
    x::Vector{T},
    g::Vector{T},
    H::GN{T},
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T},
    delta::T,
    t0::T,
    kappa1::T,
    gamma_c::T) where T

    # Bounds on the step 
    t_trial = t0
    t_small = 1e-6
    s = projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x

    increase = false
    s_infnorm = norm(s,Inf)
    if s_infnorm > delta
        increase = false
    else
        gts = dot(g,s)
        # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
        qs = 0.5*vthv(H,s) +  gts
        progress = qs <= kappa1 * gts
    end

    if increase
        t_c = t_trial
        progress = true
        while progress
            t_trial *= gamma_c
            s = projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x
            s_infnorm = norm(s,Inf)
            if s_infnorm <= delta
                gts = dot(g,s)
                # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                qs = 0.5*vthv(H,s) +  gts
                progress = qs <= kappa1 * gts
                if progress t_c = t_trial end 
            else
                progress = false
            end
        end
    else
        satisfied = false
        while !satisfied && t_trial > t_small
            t_trial /= gamma_c
            s =  projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x
            s_infnorm = norm(s,Inf)
            if s_infnorm <= delta
                gts = dot(g,s)
                # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                qs = 0.5*vthv(H,s) +  gts
                satisfied = qs <= kappa1 * gts
            end
        end
        t_c = t_trial
    end

    # This computation is (likely) uneccessary, retrieving the cauchy step from above should be possible 
    s_c = projection_polyhedron(x-t_c*g, A, b, x_l, x_u) - x
    return s_c, t_c
end


""" next_breakpoint(d,s,dₗ,dᵤ,fix_bounds)

Finds the smallest scalar `θ` such that one component not in `fix_bounds` of `s + θ*d` lies at one of the bounds `dₗ` or `dᵤ`.   

Returns the scalar `θ` and `ind`, the index of the component that becomes active.
"""
function next_breakpoint(
        d::Vector{T},
        s::Vector{T},
        d_l::Vector{T},
        d_u::Vector{T},
        fix_bounds::BitVector) where T

    theta = Inf
    ind = -1

    for i in axes(d,1)
        if !fix_bounds[i]
            if d[i] < 0
                theta_try = (d_l[i]-s[i]) / d[i]
            elseif d[i] > 0 
                theta_try = (d_u[i]-s[i]) / d[i]
            else theta_try = Inf
            end

            if theta_try < theta
                theta = theta_try
                ind = i
            end
        end
    end
    return theta, ind
end

""" cauchy_step(x,g,H,A,chol_AAᵀ,lincons)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model `q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the projected gradient step `P(x-t*g) - x` with `P` denoting the projection over `{v | Av = 0 and max(-Δ,xₗ) ≤ x + v ≤ min(Δ,xᵤ)}`.

Matrix `A` and bounds `xₗ`, `xᵤ` are encoded by the [`MixedConstraints`](@ref) object `lincons`.

This method finds the first local minimum of the quadratic model along the projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

Returns the associated Cauchy step `s_c` and `fix_bounds`, a `BitVector` encoding the indices of active bounds at `x + s_c` 
"""
function cauchy_step(
        x::Vector{T},
        g::Vector{T},
        H::GN,
        chol_aat::Cholesky{T,Matrix{T}},
        lincons::MixedConstraints{T},
        delta::T) where T

    # Dimensions and constants
    (m,n) = size(lincons.lineq)
    nmm = n-m
    atol = sqrt(eps(T))
    # Buffers
    s_c = zeros(n) # accumulated projected gradient step
    t = zero(T) # scalar to store breakpoint value

    # Initial active bounds
    active_bounds!(lincons,x,chol_aat)
    d = projection(lincons,-g)
    # d = projection(A,chol_aat,-g)
    # fix_bounds = BitVector(map(t -> abs(t) < atol, d))
    
    # if any(fix_bounds)
    #     chol_aug_aat = cholesky_aug_aat(A,fix_bounds,chol_aat)
    #     projection!(d,A,chol_aug_aat,fix_bounds,-g)
    # end
  
    # Upper and lower bounds for the Cauchy step
    d_u = (t -> min(t, delta)).(lincons.xupp-x)
    d_l = (t -> max(t, -delta)).(lincons.xlow-x)
    
    # slope and curvature on the current interval
    # chol_aug_aat = cholesky_aug_aat(A,fix_bounds,chol_aat)
    #d = projection(A, chol_aug_aat, ifix, -g)
    # d = projection(A,chol_aat,-g)
    Hd = H*d
    phi_p = dot(s_c,Hd) + dot(g,d)
    phi_pp = dot(d,Hd)
    
    min_found = false

    while !min_found && (nb_fix(lincons) < nmm)
        
        theta, ind = next_breakpoint(d,s_c,d_l,d_u,lincons.fixvars)
        delta_t = (phi_pp > 0 ? -phi_p / phi_pp : zero(T))

        if phi_p >= 0 # local minimum at t
            min_found = true
        elseif phi_p < 0 && phi_pp > 0 && delta_t < theta
            delta_t = -phi_p / phi_pp
            delta_t < theta # local minimum at t - phi_p / phi_pp
            s_c[:] += delta_t*d
            min_found = true
        else # no local minimum in [t, t+theta), prepare for the next interval 
            s_c[:] += theta*d[:]
            # fix_bounds[ind] = true
            # chol_aug_aat = cholesky_aug_aat(A,fix_bounds,chol_aat)
            add_active!(lincons,chol_aat,ind)
            projection!(lincons,-g,d)
            Hd[:] = H*d
            phi_p = dot(s_c,Hd) + dot(g,d)
            phi_pp = dot(d,Hd)
        end
    end
    return s_c
end

""" minor_iterate(x,s,g,H,A,xₗ,xᵤ,fixed_var,Δ,κ₂)

Compute a search direction `w` and a steplength `α` such that the next minor iterate `x + s + α*w` provides a sufficient reduction, where

* `x` is the current iterate 

* `s` is the previous minor step or, equivalently, `x + s` is the previous minor iterate
"""
function minor_iterate(
    x::Vector{T},
    s::Vector{T},
    g_model::Vector{T},
    H::GN{T},
    lincons::MixedConstraints{T},
    delta::T,
    kappa2::T) where T

    n = size(x,1)

    x_minor = x+s
    # free_var = free_index(fixed_var)
    w_u, w_l = fill(Inf,n), fill(-Inf,n)

    w_u[lincons.fixvars] .= (t -> min(t, delta)).(lincons.xupp[lincons.fixvars]-x_minor[lincons.fixvars])
    w_l[lincons.fixvars] .= (t -> max(t, -delta)).(lincons.xlow[lincons.fixvars]-x_minor[lincons.fixvars])

    w, cg_status = projected_cg(g_model,H,w_l,w_u,lincons,kappa2)

    if cg_status != negative_curvature
        alpha = linesearch(g_model,H,w,w_l,w_u,lincons.fixvars)
        w .= alpha * w
    end

    return w, cg_status
end

""" projected_cg(r0,H,A,wₗ,wᵤ,chol_AₓAₓᵀ,fix_bounds,κ₂,atol)

Approximately solve the sub problem 

`min 0.5 wᵀHw + wᵀr0 
s.t. Aw = 0,
     wᵢ = 0,    i ∈ fix_bounds
     wₗ ≤ w ≤ wᵤ,`

with respect to `w` and using the projected conjugate gradient method with early termination if a direction hits a bound.

Returns the obtained descent direction and the termination status, encoded in the `Enum` (see [`CG_status`](@ref))
"""
function projected_cg(
    g_minor::Vector{T},
    H::GN{T},
    w_l::Vector{T},
    w_u::Vector{T},
    lincons::MixedConstraints{T},
    kappa2::T;
    atol::T = sqrt(eps(T))) where T

    (m,n) = size(lincons.lineq)
    
    # Initialization
    w = zeros(n)
    r = zeros(n)
    # r = H*s + g
    r[:] = g_minor[:]
    v = projection(lincons,r)
    rtv = dot(r,v)
    p = -v

    tol_cg = kappa2 * norm(v)
    tol_negcurve = atol

    iter = 1
    max_iter = 2*(n-m-nb_fix(lincons))
    # approx_solved = abs(rtv) < tol_cg
    approx_solved = false
    neg_curvature = false
    outside_region = false

    while !approx_solved && !outside_region && !neg_curvature && iter <= max_iter
        # println("[projected_cg] iter ", iter)
        Hp = H*p
        pHp = dot(p,Hp)

        if pHp <= tol_negcurve # negative curvature
            neg_curvature = true
            if abs(pHp) > tol_negcurve # nonzero curvature
                gamma = factor_to_boundary(p,w,w_l,w_u)
                w .+= gamma * p
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp
            gamma = factor_to_boundary(p,w,w_l,w_u)
            outside_region = alpha > gamma
            if outside_region
                w .+= gamma * p
            else 
                w .+= alpha*p
                r .+= alpha*Hp
                projection!(lincons,r,v)
                # v = projection(A,chol_aug_aat,fix_bounds,r)
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-one(T), v, beta, p) # p = -v + βp
                rtv = rtv_next
                approx_solved = abs(rtv) < tol_cg
                iter += 1
            end
        end
    end

    status = if approx_solved
        solved
    elseif outside_region
        bound_hit
    elseif neg_curvature
        negative_curvature
    elseif iter == max_iter
        max_iter_reached
    end

    return w, status
end

function linesearch(
    g_model::Vector{T},
    H::GN{T},
    w::Vector{T},
    w_l::Vector{T},
    w_u::Vector{T},
    fix_bounds::BitVector) where T 

    # unconstrained minimizer of the model along the descent direction
    wHw = vthv(H,w)
    alpha_opt = (wHw > 0 ? -dot(g_model,w) / wHw : Inf)

    # Maximum step allowed by the bounds on the free variables
    alpha_allowed =  Inf
    for i in axes(w,1)
        if !fix_bounds[i]
            if w[i] < 0
                alpha_allowed = min(alpha_allowed, w_l[i] / w[i])
            elseif w[i] > 0
                alpha_allowed = min(alpha_allowed, w_u[i] / w[i])
            end
        end
    end

    return min(alpha_opt, alpha_allowed)
end

function factor_to_boundary(
    p::Vector{T},
    w::Vector{T},
    w_l::Vector{T},
    w_u::Vector{T};
    atol::T = sqrt(eps(T))) where T 

    gamma = Inf
    for i in axes(w,1)
        if p[i] < -atol 
            gamma = min(gamma, (w_l[i] - w[i]) / p[i])
        elseif p[i] > atol
            gamma = min(gamma, (w_u[i] - w[i]) / p[i])
        end
    end
    return gamma
end


function criticality_measure(
    g::Vector{T},
    lincons::MixedConstraints{T}) where T 

    return norm_reduced_gradient(g,lincons)
end

#= Compute the criticality measure ||P(x-∇f)-x|| where P[.] denotes the orthogonal projection on {x | Ax=b, xₗ ≤ x ≤ xᵤ}

=#

function criticality_measure(
    x::Vector{T},
    g::Vector{T},
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T}) where T

    p_xmg = projection_polyhedron(x-g,A,b,x_l,x_u)
    return norm(p_xmg-x)
end

#= Return the norm of the reduced gradient `Ñᵀg`
    
    * `g` is the gradient of the augmented Lagrangian at current iterate

    * `Ñ` is an orthonormal matrix representing the null space of current active linear constraints

=#
function norm_reduced_gradient(
    g::Vector{T},
    polyhedron::MixedConstraints) where T

    reduced_g = projection(polyhedron,-g)
    return norm(reduced_g)
end


# function norm_reduced_gradient(
#     g::Vector{T},
#     A::Matrix{T},
#     chol_aat::Cholesky{T,Matrix{T}}) where T

#     reduced_g = projection(A,chol_aat,-g)
#     return norm(reduced_g)
# end





#= Compute an approximate Cauchy point by finding the first local minimum of a piecewise quadratic path 

DOES NOT WORK 
function cauchy_point!(x::Vector{T},
                       s_gc::Vector{T},
                       ∇f::Vector{T},
                       H::Matrix{T},
                       A::Matrix{T},
                       chol_AAᵀ::Cholesky{T,Matrix{T}},
                       ℓ::Vector{T},
                       u::Vector{T},
                       Δ::T,
                       verbose::Bool=false) where T
    (m,n) = size(A)
    zero_T = zero(T)
    # Buffer initialization
    s_i, e_i = zeros(T,n), zeros(T,n)
    d, d_i, Hdi = Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n)
    
    # Orthogonal projection of ∇f onto the null space of A
    projection!(d,A,chol_AAᵀ,∇f)
    d_i[:] = d[:]
    ℓ_bar = map(t -> max(t,-Δ), ℓ - x)
    u_bar = map(t -> min(t,Δ), u - x)

    # Breakpoints
    t_b, t_b_sorted = break_points(d, ℓ_bar, u_bar)

    t_i = zero_T
    for j in findall(isapprox(t_i),t_b)
        e_i[j] = d[j]
    end
    axpy!(-1,e_i,d_i)

    gtdi = dot(∇f,d_i)
    mul!(Hdi,H,d_i)

    minima_found = false
    while !minima_found || !isempty(t_b_sorted)

        # Slope and curvature
        qi_p = gtdi + dot(s_i,Hdi)
        qi_pp = dot(d_i,Hdi)
        Δt = (isapprox(zero_T)(qi_pp) ? zero_T : -qi_p / qi_pp)
        t_ip1 = popfirst!(t_b_sorted)

        if qi_p ≥ 0
            s_gc[:] = s_i[:]
            minima_found = true
        elseif qi_pp > 0 && Δt > t_ip1 - t_i
            s_gc[:] = s_i + Δt*d_i[:]
            minima_found = true
        else # Prepare for the next interval
            j_break = findall(isapprox(t_ip1),t_b)
            Δt = t_ip1 - t_ip
            axpy!(Δt,d_i,s_i)
            e_i[:] = [(j in j_break ? d[j] : zero_T) for j=1:n]
            axpy(-1,e_i,d_i)
            # Update gᵀd and Hd
            gtdi -= dot(∇f,e_i)
            mul!(Hdi,H,e_i,-1,1)
        end
    end
    
    return
end

function break_points(d::Vector{T},l::Vector{T},u::Vector{T}) where T
    @assert all(<(0),l) && all(>(0),u) "Upper and lower bounds must be of opposite sign"
    
    t_b = zeros(T, size(d,1))
    for i ∈ axes(d,1)
        if d[i] < 0
            t_b[i] = l/d[i]
        elseif d[i] > 0
            t_b[i] = u/d[i]
        end
    end
    # Ordered values
    t_b_sorted = sort(unique(t_b))
    filter(!isapprox(zero(T),t_b_sorted))  # Get rid of zero value breakpoint for convenience
    
    return t_b, t_b_sorted
end

=#
