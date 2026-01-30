# ANALACNLS : A New Augmented Lagrangian Algorithm for Constrained Nonlinear Least-Squares

function analacnls(
    x0::Vector{T},
    res::F1,
    nlcons::F2,
    jac_res::F3,
    jac_nlcons::F4;
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
    kappa_sfg::T = T(1e-7),
    kappa_smlres::T = T(0.1),
    max_outer_iter::Int = 500,
    max_inner_iter::Int = 500,
    verbose::Bool=false) where {T<:Real, F1<:Function, F2<:Function, F3<:Function, F4<:Function}

    # Initial sanity checks
    @assert (0 < eta1 <= eta2 < 1) && (0 < gamma1 < 1 < gamma2) "Invalid trust region updates paramaters"

    n = size(x0,1)
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    mu = mu0

    omega, eta = initial_tolerances(mu0, omega0, eta0, k_crit, k_feas) # tolerances 
    y = least_squares_multipliers(x,res,jac_res,jac_nlcons) # Initial Lagrange multipliers 

    feas_measure = Inf
    pix = Inf

    first_order_critical = false
    outer_iter = 1

    while !first_order_critical && outer_iter <= max_outer_iter

        verbose && println("Outer iter $outer_iter")

        x_next, cx_next, pix = solve_subproblem(
            x,
            y,
            mu,
            res,
            nlcons,
            jac_res,
            jac_nlcons,
            omega,
            max_inner_iter,
            eta1,
            eta2,
            gamma1,
            gamma2,
            kappa_sfg,
            kappa_smlres)

        

        feas_measure = norm(cx_next)
        verbose && @show feas_measure, pix

        if feas_measure <= eta
            x .= x_next
            cx = cx_next
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
    end
    return x,y, feas_measure, pix
end

function solve_subproblem(
    x0::Vector{T},
    y::Vector{T},
    mu::T,
    res::F1,
    nlcons::F2,
    jac_res::F3,
    jac_nlcons::F4,
    omega_subpb::T,
    iter_max::Int,
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T,
    kappa_sfg::T,
    kappa_smlres::T;
    verbose::Bool=false) where {T, F1<:Function, F2<:Function, F3<:Function, F4<:Function}

    # Buffers initializations
    n = size(x0,1)
    x, x_next = Vector{T}(undef,n), Vector{T}(undef,n)
    g = Vector{T}(undef,n)
    
    x[:] .= x0[:]
    rx, cx, mx = evaluate_al(x,y,mu,res,nlcons)
    y_bar, Jx, Cx, g = first_derivatives(x,y,mu,rx,cx,jac_res,jac_nlcons)
    H = GN(Jx,Cx,mu) # start with Gauss-Newton approximation 
    S = zeros(n,n)
    delta = initial_tr(g)
    k = 1
    pix = norm(g)
    optimal = pix < omega_subpb

    while !optimal && k <= iter_max
        verbose && println("[solve_subproblem] inner iter $k")
        s = dogleg_step(H,g,delta)
        
        x_next[:] .= (x[:] .+ s[:])
        rx_next, cx_next, mx_next = evaluate_al(x_next,y,mu,res,nlcons)

        # ratio 
        pred = dot(g,s) + 0.5*vthv(H,s)
        ared = mx_next - mx
        rho = ared / pred
        # Good step 
        if rho > eta1
            x[:] .= x_next[:] 
            # update quadratic model 
            y_bar_next, Jx_next, Cx_next, g_next = first_derivatives(x,y,mu,rx_next,cx_next,jac_res,jac_nlcons)
            
            # Gauss-Newton update 
            H = GN(Jx_next,Cx_next,mu)
            # SR1 update 
            # H, S[:,:] = update_hessian(s,g_next,rx_next,Jx_next,Cx_next,mu,y_bar_next, Jx, Cx, S, kappa_sfg)
            # verbose && @show S
            # hybrid version
            # H, S[:,:] = hybrid_update_hessian(s,mx,mx_next,g_next,rx_next,Jx_next,Cx_next,mu,y_bar_next,Jx,Cx,S,kappa_smlres,kappa_sfg)

            rx[:], cx[:], mx = rx_next[:], cx_next[:], mx_next
            g[:], y_bar[:] = g_next[:], y_bar_next[:]
            Jx[:,:], Cx[:,:] = Jx_next[:,:], Cx_next[:,:]

        end

        # update radius 
        delta = update_tr(delta, rho, eta1, eta2, gamma1, gamma2)
        # test convergence criteria
        pix = norm(g) 
        optimal = pix < omega_subpb
        k += 1
    end
    return x, cx, pix
end

function newton_step(H::AlSR1Hessian{T},g::Vector{T}) where T
    # explicity form of the Hessian 
    # temporary, will be improved if the first benchmarks are promising 
    M = H.J'*H.J + H.mu*H.C'*H.C + H.S 
    return M \ g
end 

function newton_step(H::GN{T}, g::Vector{T}) where T 
    # explicity form of the Hessian 
    # temporary, will be improved if the first benchmarks are promising 
    M = H.J'*H.J + H.mu*H.C'*H.C
    return M \ g
end



function dogleg_step(
    H::ALHessian,
    g::Vector{T},
    delta::T
) where T 

    s = zeros(T,size(g,1))
    Hg = H*g 
    s_c = -(dot(g,g) / dot(g,Hg)) * g # Cauchy point 

    # Temporary: only take the Cauchy point
    if norm(s_c) >= delta
        s[:] .= -(delta / norm(g)) *g
    else
        s[:] = s_c[:]
    end

    # if norm(s_c) >= delta
    #     s[:] .= -(delta / norm(g)) *g
    # else
    #     s_n = newton_step(H,g) # computes the Newton step H⁻¹ g 
    #     norm_sn = norm(s_n)
    #     if  norm_sn > delta
    #         d = s_n-s_c
    #         alpha = factor_to_boundary(s_c,d,delta)
    #         s[:] .= s_c + alpha*d
    #     else
    #         s[:] .= s_n[:]
    #     end
    # end

    return s
end
