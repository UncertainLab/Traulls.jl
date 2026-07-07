# Problem 14 from Hock-Schittkowski

function hs14()

    n = 2
    m = 2
    p = 1

    r(x) = [x[1] - 2.0, x[2] - 1]

    jac_r(x) = [1.0 0.0;
    0.0 1.0]

    g(x) = -0.25 * x[1]^2 - x[2]^2 + 1

    jac_g(x) = [-0.5*x[1] -2x[2]]

    A = [1.0 -2.0]
    b = [-1.0]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = 2*ones(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, A, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end