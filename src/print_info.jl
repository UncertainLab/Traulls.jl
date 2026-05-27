function print_traulls_header(
    model::AbstractCnlsModel{T},
    fx::T,
    feas_measure::T,
    pix::T,
    omega_rel::T,
    feas_tol::T,
    tau::T,
    tr::TrustRegion{T};
    io::IO=stdout) where T

    print(io,"\n\n")
    println(io, '*'^80)
    println(io, "*",' '^78,"*")

    println(io, "*"," "^31,"Traulls.jl v-DEV"," "^31,"*")
    println(io, "*",' '^78,"*")
    println(io, "*       Trust Region Augmented Lagrangian nonLinear Least-squares Solver       *")
    println(io, "*",' '^78,"*")
    println(io, '*'^80)

    # Problem information
    println(io, model)

    println(io, "\nAlgorithm parameters")
    println(io, "Relative criticality tolerance.......................: ", @sprintf("%.6e", omega_rel))
    println(io, "Feasibility tolerance for equality constraints.......: ", @sprintf("%.6e", feas_tol))
    println(io, "Increase penalty parameter factor....................: ", @sprintf("%5f", tau))

    # Trust region parameters
    println(io, tr)

    # Initial optimality quantities
    println(io, "\nAt starting point: ")
    println(io, "Objective...................: ", @sprintf("%.6e", fx))
    println(io, "Nonlinear feasibility.......: ", @sprintf("%.6e", feas_measure))
    println(io, "Criticality measure.........: ", @sprintf("%.6e", pix))


    # Iteration detail header
    println(io,"\n",'='^37, " Iteration detail ", '='^37)
    println(io,"\niter    objective    nl feasibility      μ      criticality        status")
end

function print_outer_iteration(
    iter::Int,
    objective::Float64,
    nl_feas::Float64,
    mu::Float64,
    pix::Float64,
    ::Val{true};
    io::IO=stdout)

    @printf(io, "%4d  %.7e   %.6e   %.2e     %.2e    update multipliers\n", iter, objective, nl_feas, mu, pix)

end

function print_outer_iteration(
    iter::Int,
    objective::Float64,
    nl_feas::Float64,
    mu::Float64,
    pix::Float64,
    ::Val{false};
    io::IO=stdout)

    @printf(io, "%4d  %.7e   %.6e   %.2e     %.2e    increase penalty\n", iter, objective, nl_feas, mu, pix)

end

function print_inner_iter_header(io::IO=stdout)
    println(io,"\n",'='^92)
    println(io,"iter      AL value       ||s||        Δ           ρ")
end

function print_inner_iter(
    iter::Int,
    obj::Float64,
    norm_step::Float64,
    radius::Float64,
    rho::Float64;
    io::IO=stdout) 

    @printf(io, "%4d   %13.6e   %.2e   %.2e   %9.2e\n", iter, obj, norm_step, radius, rho)
    return
end
