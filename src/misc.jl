function print_tralcnllss_header(
    n::Int,
    d::Int,
    p::Int,
    m::Int,
    x_l::Vector{Float64},
    x_u::Vector{Float64},
    crit_tol::Float64,
    feas_tol::Float64,
    tau::Float64,
    eta1::Float64,
    eta2::Float64,
    gamma1::Float64,
    gamma2::Float64;
    io::IO=stdout) 

    print(io,"\n\n")
    println(io, '*'^64)
    println(io, "*",' '^62,"*")

    println(io, "*"," "^23,"BEnlsip.jl v-DEV"," "^23,"*")
    println(io, "*",' '^62,"*")
    println(io, "*                   Better version of ENLSIP                   *")
    println(io, "*",' '^62,"*")
    println(io, '*'^64)

    println(io, "\nProblem dimensions")
    println(io, "Number of parameters.................: ", @sprintf("%5i", n))
    println(io, "Number of residuals..................: ", @sprintf("%5i", d))
    println(io, "Number of nonlinear constraints......: ", @sprintf("%5i", p))
    println(io, "Number of linear constraints.........: ", @sprintf("%5i", m))
    println(io, "Number of lower bounds...............: ", @sprintf("%5i", count(isfinite, x_l)))
    println(io, "Number of upper bounds...............: ", @sprintf("%5i", count(isfinite, x_u)))
    println(io, "\nAlgorithm parameters")
    println(io, "Relative criticality tolerance.......................: ", @sprintf("%.6e", crit_tol))
    println(io, "Nonlinear constraints feasibility tolerance..........: ", @sprintf("%.6e", feas_tol))
    println(io, "Increase penalty parameter factor....................: ", @sprintf("%5f", tau))
    println(io, "Step acceptance treshold.............................: ", @sprintf("%5f", eta1))
    println(io, "Great step acceptance treshold.......................: ", @sprintf("%5f", eta2))
    println(io, "Trust region increase factor.........................: ", @sprintf("%5f", gamma2))
    println(io, "Trust region decrease factor.........................: ", @sprintf("%5f", gamma1))
    println(io,"\n\n")

    return
end

function print_boconls_header(
    n::Int,
    m::Int,
    p::Int,
    x_l::Vector{Float64},
    x_u::Vector{Float64},
    omega_rel::Float64,
    feas_tol::Float64,
    tau::Float64;
    io::IO=stdout) 

    print(io,"\n\n")
    println(io, '*'^64)
    println(io, "*",' '^62,"*")

    println(io, "*"," "^23,"BEnlsip.jl v-DEV"," "^23,"*")
    println(io, "*",' '^62,"*")
    println(io, "*                   Better version of ENLSIP                   *")
    println(io, "*",' '^62,"*")
    println(io, '*'^64)

    println(io, "\nProblem dimensions")
    println(io, "Number of parameters.................: ", @sprintf("%5i", n))
    println(io, "Number of residuals..................: ", @sprintf("%5i", m))
    println(io, "Number of equality constraints.......: ", @sprintf("%5i", p))
    println(io, "Number of lower bounds...............: ", @sprintf("%5i", count(isfinite, x_l)))
    println(io, "Number of upper bounds...............: ", @sprintf("%5i", count(isfinite, x_u)))
    println(io, "\nAlgorithm parameters")
    println(io, "Relative criticality tolerance.......................: ", @sprintf("%.6e", omega_rel))
    println(io, "Feasibility tolerance for equality constraints.......: ", @sprintf("%.6e", feas_tol))
    println(io, "Increase penalty parameter factor....................: ", @sprintf("%5f", tau))
    println(io,"\n")

    return
end

function print_outer_iter_header(
    iter::Int,
    objective::Float64,
    nl_feas::Float64,
    mu::Float64,
    pix::Float64,
    omega::Float64;
    io::IO=stdout) 

    println(io,"\n",'='^80)
    println(io,"                          Iter $iter")
    println(io,"  objective    nl feasibility      μ      criticality   tolerance")
    if iter == 1
        @printf(io, "%.7e   %.6e   %.2e        -         %.2e", objective, nl_feas, mu, omega)
    else
        @printf(io, "%.7e   %.6e   %.2e     %.2e     %.2e", objective, nl_feas, mu, pix, omega)
    end
    println(io,"\n",'='^80)
    println(io,"iter     AL value       ||s||        Δ          ρ")
    return
end

function print_inner_iter(
    iter::Int,
    obj::Float64,
    norm_step::Float64,
    radius::Float64,
    rho::Float64;
    io::IO=stdout) 

    @printf(io, "%4d   %.6e   %.2e   %.2e   %.2e\n", iter, obj, norm_step, radius, rho)
    return
end

function print_termination_info(
    iter::Int,
    x::Vector,
    y::Vector,
    mu::Float64,
    obj::Float64,
    criticality::Float64,
    feasibility::Float64;
    io::IO=stdout)

    println(io,"\n",'='^80)
    println(io, "\nTerminated after $(iter-1) outer iterations\n")
    println(io, "Squared sum of residuals............................: ", @sprintf("%.6e", obj))
    println(io, "Criticality measure.................................: ", @sprintf("%.6e", criticality))
    println(io, "Feasibility of equality constraints.................: ", @sprintf("%.6e", feasibility))
    println(io, "Final value of the penalty parameter................: ", @sprintf("%.3e", mu))

    println(io, "\nPrimal solution...................................")
    (t -> @printf(io, " %.7e ",t)).(x)
    println(io, "\n\nLagrange multipliers............................")
    (t -> @printf(io, " %.7e ",t)).(y)

    return
end
