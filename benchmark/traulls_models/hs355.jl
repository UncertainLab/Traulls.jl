# Problem 355 from Hock Schittkowski collection

function hs355()

    n = 4
    m = 2
    p = 1

    # Residuals
    r(x) = [11.0 - x[1]*x[4] - x[2]*x[4] + x[3]*x[4],
            x[1] + 10*x[2] - x[3] + x[4] + x[2]*x[4]*(x[3]-x[1])]

    jac_r(x) = ForwardDiff.jacobian(r,x)

    # Constraints
    c(x) = [(11.0 - x[1]*x[4] - x[2]*x[4] + x[3]*x[4])^2 + (x[1] + 10*x[2] - x[3] + x[4] + x[2]*x[4]*(x[3]-x[1]))^2 -
        (11.0 - 4*x[1]*x[4] - 4*x[2]*x[4] + x[3]*x[4])^2 - (2*x[1]+20*x[2] -0.5*x[3] + 2*x[4] + 2*x[2]*x[4]*(x[3]-4*x[1]))^2]

    jac_c(x) = ForwardDiff.jacobian(c,x)

    x_low = [0.1, 0.1, 0.0, 0.0]
    x_upp = fill(Inf,n)

    # Starting point
    x_sol = [1.917, 0.1, 0.0, 1.972]
    x = zeros(n)

    return Traulls.CnlsModel(r,c,jac_r,jac_c,x_low,x_upp,x,n,m,p,Val(:only_equalities))
end
