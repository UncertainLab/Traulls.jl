# Problem 22 from Hock-Schittkowski

function hs22()

    # Additional slack variable for the linear constraints
    n = 3
    m = 2
    p = 1

    r(x) = [x[1] - 2.0, x[2] - 1.0]

    jac_r(x) = [1.0 0.0 0.0; 
    0.0 1.0 0.0]

    g(x) = -x[1]^2 + x[2]

    jac_g(x) = [-2x[1] 1.0 0.0]

    A = [-1.0 -1.0 -1.0]
    b = [-2.0]

    xlow = [-Inf, -Inf, 0.0]
    xupp = fill(Inf, n)

    x0 = [2.0, 2.0, 0.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, A, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end