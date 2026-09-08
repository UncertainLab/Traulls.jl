# Problem 15 from Hock-Schittkowski

function hs15()

    n = 2
    m = 2
    p = 2

    r(x) = [10 * (x[2] - x[1]^2), 1.0 - x[1]]

    jac_r(x) = [-20*x[1] 10.0;
                -1.0 0.0]

    g(x) = [x[1]*x[2] - 1.0, 
            x[1] + x[2]^2]

    jac_g(x) = [x[2] x[1];
                1.0 2x[2]]

    xlow = [-Inf, -Inf]
    xupp = [0.5, Inf]

    x0 = [-2.0, 1.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end