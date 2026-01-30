

# Trust Region AUgmented Lagrangian solver for constrained nonlinear Least-Squares

# Structure encoding a nonlinear least-squares problems subject to nonlinear equality constraints, linear equality constraints and bound
# constraints.
mutable struct PolyhedralCnls <: AbstractCnlsModel
    res
    nleq
    nlineq
    jac_res
    jac_nleq
    jac_nlineq
    eqmat::Matrix
    eqrhs::Vector
    x_low::Vector
    x_upp::Vector
    n::Int
    n_slack::Int
    m::Int
    p::Int
end

#= Methods to evaluate residuals, nonlinear constraints and jacobians of a given model
Methods are implemented in both in place and out of place versions 
In place versions are implemented in local scope inside the solver =#



"""
    residuals!(model::PolyhedralCnls, x::Vector, v::Vector)

Compute the residuals for the given model and input vector `x`, storing the result in `v`.
"""
function residuals!(model::PolyhedralCnls, x::Vector, v::Vector) end

"""
    residuals(model::PolyhedralCnls, x::Vector) 

Return the residuals for the given model and input vector `x` as a new vector.
"""
function residuals(model::PolyhedralCnls,x::Vector)  
    rx = Vector{eltype(x)}(undef,model.m)
    residuals!(model, x, rx)
    return rx
end

"""
    nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) 

Compute the nonlinear constraints for the given model and input vector `x`, storing the result in `v`.
"""
function nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) end

"""
    nlconstraints(model::PolyhedralCnls, x::Vector) 

Return the nonlinear constraints for the given model and input vector `x` as a new vector.
"""
function nlconstraints(model::PolyhedralCnls,x::Vector)  
    cx = Vector{eltype(x)}(undef,model.p)
    nlconstraints!(model, x, cx)
    return cx
end

"""
    jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) 

Compute the Jacobian of the residuals for the given model and input vector `x`, storing the result in matrix `J`.
"""
function jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) end

"""
    jac_residuals(model::PolyhedralCnls, x::Vector) 

Return the Jacobian of the residuals for the given model and input vector `x` as a new matrix.
"""
function jac_residuals(model::PolyhedralCnls, x::Vector)  
    Jx = Matrix{eltype(x)}(undef,model.m, model.n)
    jac_residuals!(model, x, Jx)
    return Jx
end

"""
    jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix) 

Compute the Jacobian of the nonlinear constraints for the given model and input vector `x`, 
storing the result in matrix `C`.
"""
function jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix) end

"""
    jac_nlconstraints(model::PolyhedralCnls, x::Vector) 

Return the Jacobian of the nonlinear constraints for the given model and input vector `x` as a new matrix.
"""
function jac_nlconstraints(model::PolyhedralCnls,x::Vector)  
    Cx = Matrix{eltype(x)}(undef,model.p,model.n)
    jac_nlconstraints!(model, x, Cx)
    return Cx
end

# Main solving method 

function solve(
    model::PolyhedralCnls;
    x::Vector{Float64}=zeros(model.n),
    mu::Float64 = 10.0,
    tau::Float64 = 10.0,
    omega0::Float64 = 1.0,
    eta0::Float64 = 1.0,
    feas_atol::Float64 = 1e-6,
    crit_rtol::Float64 = 1e-7,
    k_crit::Float64 = 1.0,
    k_feas::Float64 = 0.1,
    beta_crit::Float64 = 1.0,
    beta_feas::Float64 = 0.9,
    accept_treshold::Float64 = 0.25,
    increase_treshold::Float64 = 0.75,
    decrease_factor::Float64 = 0.5,
    increase_factor::Float64 = 2.5,
    neg_ratio_factor::Float64 = 0.0625,
    kappa_step::Float64 = 0.1,
    kappa_cg::Float64 = 0.1,
    kappa_sos::Float64 = sqrt(eps(Float64)),
    kappa_sml_res::Float64 = 0.1,
    hessian_approx::HessianApprox = gn,
    max_iter::Int = 100,
    max_inner_iter::Int = 100,
    max_cg_iter::Int = 50,
    output_file_name::String="",
    verbose::Bool=false)

    
    # Sanity check on parameters
    @assert (0 < accept_treshold <= increase_treshold < 1) && (0 < decrease_factor < 1 < increase_factor) "Invalid trust region paramaters"

    # Local scope evaluation methods 

    function residuals!(model::PolyhedralCnls, x::Vector, v::Vector) 
        v[:] .= model.res(x[1:model.n-model.n_slack])
        return 
    end

    function nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) 
        n, n_slack, p = model.n, model.n_slack, model.p
        n_var = n - n_slack
        p_eq = p - n_slack

        x_var = view(x,1:n_var)
        x_slack = view(x,n_var+1:n)

        # Equality constraints components
        if p_eq > 0 v[1:p_eq] .= model.nleq(x_var) end 

        # Inequality constraints transformed into equalities
        if n_slack > 0  v[p_eq+1:end] .= model.nlineq(x_var) .- x_slack end

        return
    end

    function jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) 
        n, n_slack, m = model.n, model.n_slack, model.m
        n_var = n - n_slack

        J[:,1:n_var] .= model.jac_res(x[1:n_var])

        if n_slack > 0
            J[:,n_var+1:end] .= zeros(m,n_slack)
        end
        return
    end

    function jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix)
        n, n_slack, p = model.n, model.n_slack, model.p
        n_var = n - n_slack
        p_eq = p-n_slack

        x_var = view(x,1:n_var)

        # Equality constraints components
        if p_eq > 0 
            C[1:p_eq,:] .= hcat(model.jac_nleq(x_var), zeros(p_eq,n_slack)) 
        end

        # Inequality constraints transformed into equalities
        if n_slack > 0 
            C[p_eq+1:end,:] .= hcat(model.jac_nlineq(x_var), Diagonal{Float64}(-I,n_slack)) 
        end

        return
    end


    n, m, p = model.n, model.m, model.p
    x_low, x_upp = model.x_low, model.x_upp

    A = model.eqmat
    chol_aat = cholesky(A*A')

    # Initialize structures
    tr = TrustRegion(accept_treshold, increase_treshold, decrease_factor, increase_factor, neg_ratio_factor)
    proj_op = SubspaceProjector(A,chol_aat)

    # Output stream
    output_stream = output_file_name == "" ? stdout : output_file_name

    verbose && print_boconls_header(n,m,p,x_low,x_upp,omega_rel,feas_tol,tau; io=stream)
    verbose && print_tr_header(tr;io=output_file)

    # Allocation of buffers and first evaluations
    rx = residuals(model, x)
    cx = nlconstraints(model, x)
    J = jac_residuals(model, x)
    C = jac_nlconstraints(model, x)
   
    omega_rel, eta = initial_tolerances(mu, omega0, eta0, k_crit, k_feas)  # Initial tolerances 
    y = least_squares_multipliers(rx, J, C)                            # Initial Lagrange multipliers 

    fx = dot(rx,rx)
    feas_measure = norm(cx,Inf)
    # feas_measure = norm(cx)

    g = al_grad(rx,cx,y,mu,J,C)
    g0 = copy(g) # copy initial gradient for termination criteria

    # TODO: compute a more precise initial criticality measure
    pix = criticality_measure(g0,proj_op)
    first_order_critical = pix <= crit_rtol

    iter = 1


    while !first_order_critical && iter <= max_iter

        verbose && print_outer_iter_header(iter,fx,feas_measure,mu,pix,omega; io=output_stream)

        pix = solve_subproblem(
            model,
            x,
            A,
            chol_aat,
            x_low,
            x_upp,
            proj_op,
            y,
            mu,
            rx,
            cx,
            J,
            C,
            g,
            tr,
            omega_rel,
            kappa_step,
            kappa_cg,
            kappa_sos,
            kappa_sml_res,
            hessian_approx,
            max_inner_iter,
            max_cg_iter;
            verbose=verbose,
            io=output_stream)

        feas_measure = norm(cx,Inf)

        if feas_measure <= eta
            
            pix0 = criticality_measure(g0,proj_op)
            crit_tol = max(crit_rtol, crit_rtol*pix0)
            first_order_critical = feas_measure <= feas_atol && pix <= crit_tol

            first_order_multipliers!(y,cx,mu)

            if !first_order_critical
                # Update the iterate, multipliers and decrease tolerances (penalty parameter is unchanged)
                omega = max(omega / mu^beta_crit, crit_rtol)
                eta = max(eta / mu^beta_feas, feas_atol)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances (iterate and multipliers are unchanged)
            mu *= tau
            omega = max(omega0 / mu^k_crit, crit_rtol)
            eta = max(eta0 / mu^k_feas, feas_atol)
        end

        iter += 1
        fx  = dot(rx,rx)
    end

    verbose && print_termination_info(iter,x,y,mu,fx,pix,feas_measure;io=stream)
    verbose && close(output_stream)

    PrimalDualSolution(x, y, fx, pix, feas_measure)

    end

# Function that solves the subproblem of the tralconls method 

function solve_subproblem(
    model::PolyhedralCnls,
    x::Vector,
    A::Matrix,
    chol_aat::Cholesky,
    x_low::Vector,
    x_upp::Vector,
    proj_op::SubspaceProjector,
    y::Vector,
    mu::Float64,
    rx::Vector,
    cx::Vector,
    J::Matrix,
    C::Matrix,
    g::Vector,
    tr::TrustRegion,
    omega_rel::Float64,
    kappa_step::Float64,
    kappa_cg::Float64,
    kappa_sos::Float64,
    kappa_sml_res::Float64,
    hessian_approx::HessianApprox,
    max_iter::Int,
    max_cg_iter::Int;
    verbose::Bool=false,
    io::IO=stdout)

    # Set dimensions and buffers 
    n, n_slack, p = model.n, model.n_slack, model.p
    x_prev, rx_prev, cx_prev = copy(x), copy(rx), copy(cx)
    
    reset_projector!(proj_op,chol_aat)  # set all bounds as inactive

    # Evaluate objective, first derivatives and Hessian of the AL at current point (x,y)
    alx = al_objgrad!(rx,cx,y,mu,J,C,g)
    g0 = copy(g) # save initial gradient for relative termination criteria

    H = @match hessian_approx begin
        $gn     => GN(J,C,mu)
        $sr1    => HybridSR1(J,C,mu)
    end

    set_initial_radius!(tr,g)

    solved, short_circuit = false, false 

    iter = 1 

    while !solved && iter <= max_iter && !short_circuit

        x_prev .= x 
        rx_prev .= rx 
        cx_prev .= cx
        alx_prev = alx

        radius = tr.radius 

        s, pred = projected_gradient(
            x,
            g,
            H,
            A,
            proj_op,
            x_low,
            x_upp,
            radius,
            max_cg_iter,
            kappa_step,
            kappa_cg)

        # Trial point undistinguishable from current solution or too small radius
        
        short_circuit = check_stalling(s,x,radius)

        if short_circuit continue end

        # Evaluate and analyze the reduction at trial point x+s
        x .+= s
        residuals!(model,x,rx); nlconstraints!(model,x,cx)
        alx = al_obj(rx,cx,y,mu)

        # Step taken on the slack variables, if any
        if n_slack > 0
            slack_idx = n - n_slack + 1 : n
            ineq_idx = p - n_slack + 1 : p
            
            step_slack!(x,y,cx,mu,n_slack,p)
            s[slack_idx] .= x[slack_idx] .- x_prev[slack_idx] .- s[slack_idx] # Adjust the step 
            cx[ineq_idx] .-= s[slack_idx] # Update the constraints involving slack variables without evaluating
            
            # Add reduction of the true objective function after taking second step to pred  
            pred -= alx
            alx = al_obj(rx,cx,y,mu)
            pred += alx

        end

        ratio = step_ratio(alx_prev, alx, pred)
        # verbose && println("[solve_subproblem] ared = $ared, pred = $pred, ratio = $ratio")

        if accept_step(tr,ratio)
            
            # Update the Hessian 

            if hessian_approx == gn # Gauss-Newton case
                jac_residuals!(model,x,J); jac_nlconstraints!(model,x,C) # Implicitly modifies J and C fields in H
                al_grad!(rx,cx,y,mu,J,C,g)
            
            else # Quasi Newton update
                # Form right handside of the structured secant equation
                H.secant_rhs .= -J'*rx .- C'*(y .+ mu.*cx)
                jac_residuals!(model,x,J); jac_nlconstraints!(model,x,C) # Implicitly modifies J and C fields in H
                al_grad!(rx,cx,y,mu,J,C,g)
                H.secant_rhs .+= g
                update_hessian!(H,s,alx_prev,alx,kappa_sos,kappa_sml_res)
            end

            #update_hessian!(H,J,C)
            pix = criticality_measure(g,proj_op)
            

        else
            x .= x_prev
            rx .= rx_prev
            cx .= cx_prev
            alx = alx_prev
        end

        norm_step = norm(s,Inf)
        # norm_step = norm(s)
        update_radius!(tr,ratio,norm_step)

        verbose && print_inner_iter(iter,alx_prev,norm_step,radius,ratio;io=io)

        solved = begin
            pix0 = criticality_measure(g0,proj_op)
            pix <= max(omega_rel, omega_rel*pix0)
        end

        iter += 1
    end

    return pix
end

# Computes a step by approximaelty solving a QP with the projected gradient method 
function projected_gradient(
    x::Vector,
    g::Vector,
    H::ALHessian,
    A::Matrix,
    proj_op::SubspaceProjector,
    chol_aat::Cholesky,
    x_low::Vector,
    radius::Float64,
    max_cg_iter::Int,
    kappa_step::Float64,
    kappa_cg::Float64)


    (m,n) = size(A)
    max_fixed_bounds = n-m

    s_low,s_upp = step_bounds(x,x_low,x_upp,radius)
    w_low, w_upp = Vector{Float64}(undef,n), Vector{Float64}(undef,n)

    s = cauchy_step(s,g,H,A,proj_op,chol_aat,x_low,x_upp,radius)
    Hs = H*s 
    cg_rhs = Hs .+ g 

    optimal, cg_stop = false, false
    iter = 1

    while !optimal && !cg_stop && iter <= max_cg_iter && nb_fixed(proj_op) < max_fixed_bounds

        w_low .= s_low .- s 
        w_upp .= s_upp .- s
        
        w, cg_status = pcg(cg_rhs, H, proj_op, w_low, w_upp, kappa_cg)

        s .+= w 
        Hs .= H*s 
        cg_rhs = Hs .+ g 

        # Compute norms of reduced gradients ||Zᵀg|| and ||Zᵀ(Hs+g)||
        norm_reduced_g = norm_reduced_v(g, proj_op)
        norm_reduced_gnext = norm_reduced_v(b, proj_op)

        # Evaluate termination criteria 
        optimal = norm_reduced_gnext <= kappa_step * norm_reduced_g
        cg_stop = cg_status == negative_curvature

        # Update the set of fixed variables and the projection operator 'proj_op'
        active_bounds!(s,P,chol_aat,s_low,s_upp)

        iter += 1
    end

    # Predicted reduction of the model taking step s 
    pred = dot(g,s) + 0.5*dot(s,Hs)

    return s, pred
end

""" next_breakpoint(d,s,dₗ,dᵤ,fix_bounds)

Finds the smallest scalar `θ` such that one or more components not in `fix_bounds` of `s + θ*d` lie at one of their bounds `dₗ` or `dᵤ`.   

Returns the scalar `θ` and `idx`, the index of the components that becomes active.
"""
function next_breakpoint(
        d::Vector{T},
        s::Vector{T},
        d_l::Vector{T},
        d_u::Vector{T},
        fix_bounds::BitVector;
        atol::T=sqrt(eps(T))) where T

    theta = Inf
    idx = []

    for i in axes(d,1)
        if !fix_bounds[i]
            if d[i] < -atol
                theta_try = (d_l[i]-s[i]) / d[i]
            elseif d[i] > atol 
                theta_try = (d_u[i]-s[i]) / d[i]
            else theta_try = Inf
            end

            also_bp = abs(theta_try-theta) < atol 
            
            if also_bp
                push!(idx,i)

            elseif !also_bp && theta_try < theta
                theta = theta_try
                idx = [i]
            end
        end
    end
    return theta, idx
end

# Assert if the first breakpoint is zero


#=

Reworked version of the Cauchy step computation 

Computes the step corresponding to the first local minimum of the quadratic model along the projected gradient path.

# On return

- `s`: Cauchy step  
- `P`: projection operator of type [`SubspaceProjector`](@ref) that encodes the active bounds after taking the Cauchy step
=#

""" 
    cauchy_step(x,g,H,A,chol_AAᵀ,xₗ,xᵤ,lincons)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model `q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the projected gradient step `P(x-t*g) - x` with `P` denoting the projection over `{v | Av = 0 and max(-Δ,xₗ) ≤ x + v ≤ min(Δ,xᵤ)}`.

This method finds the first local minimum of the quadratic model along the projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

# On return
- `s`: Cauchy step  
- `P`: projection operator of type [`SubspaceProjector`](@ref) that encodes the active bounds after taking the Cauchy step

""" 
function cauchy_step(
    x::Vector,
    g::Vector,
    H::ALHessian,
    A::Matrix,
    chol_aat::Cholesky,
    P::SubspaceProjector,
    x_low::Vector,
    x_upp::Vector,
    radius::Float64)

    (m,n) = size(A)
    max_fixed_bounds = n-m
    
    # Buffers 
    s = zeros(n)                    # accumulated Cauchy step 
    d = Vector{Float64}(undef,n)    # projected search direction 
    
    mul!(d,P,-g)

    prev_tb = 0
    initial_fixed = initial_active_bounds(x,d,x_low,x_upp)
    
    if !isempty(initial_fixed)
        # TODO: implement the version that does not require `chol_aat`
        add_active_bounds!(P,initial_fixed,chol_aat) 
    end

    # Update the projection 
    mul!(d,P,-g)

    # Upper and lower bounds for the projected gradient 
    d_upp = (t -> min(t, radius)).(x_upp-x)
    d_low = (t -> max(t, -radius)).(x_low-x)

    # Prepare the first interval 
    tb, idx = next_breakpoint(d,s,d_low,d_upp,P.fixvars)
    gtd = dot(g,d)
    Hd = H*d
    
    found = false

    while !found && nb_fixed(P) < max_fixed_bounds

        # Compute slope and curvature 
        phi_p = gtd + dot(s,Hd)
        phi_pp = dot(d,Hd)

        # Study the current interval [prev_tb, tb) 
        delta_t = (phi_pp > 0 ? -phi_p / phi_pp : 0.0)
        l_interval = tb - prev_tb

        if phi_p >= 0 
            # local minimum at previous breakpoint
            found = true 
        elseif phi_pp > 0 && delta_t < l_interval    
            # local minimum at t = tb - phi_p / phi_pp
            s .+= delta_t .* d 
            found = true
        else 
            # No local minimum in [prev_tb, tb) 
            # Update accumulated step
            s .+= d .* l_interval
            
            # Compute the projected direction on the next interval
            add_active_bounds!(P,idx,chol_aat)
            mul!(d,P,-g)

            # Prepare for the next interval
            gtd = dot(g,d)
            Hd = H*d 
            
            prev_tb = tb
            tb, idx = next_breakpoint(d,s,d_low,d_upp,P.fixvars)
        end

    end

    return s
end


"""
    norm_reduced_v(v,P)

Computes the norm of the reduced vector `Zᵀv` where `Z` is a null space matrix of the set `{v | Av = 0, vᵢ = 0 for i ∈ fix_vars}`
Typically `v` is the gradient of some objective function and the norm of the reduced gradient is involed to evaluate termination criteria.

# Arguments

- `v`: vector whose norm is computed
- `P`: `SubspaceProjector` operator to compute the projection of `v` onto the nullspace of interest
"""
norm_reduced_v(v::Vector,P::SubspaceProjector) = norm(P*v)

# Criticality measure for the traulls algorithm 
# Norm of the negative gradient on the subspace spanned by active constraints (all equalities + actives bounds)

function criticality_measure(
    g::Vector{T},
    proj_op::SubspaceProjector{T};
    p::T=T(Inf)) where T
    
    return norm(proj_op*(-g), p)
end

