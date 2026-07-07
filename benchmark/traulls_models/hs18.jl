# Problem 18 from Hock-Schittkowski

function hs18()
    
    n = 2
    m = 2
    p = 2

    r(x) = [0.1 * x[1], x[2]]

    jac_r(x) = [0.1 0.0;
    0.0 1.0]

    g(x) = [x[1]*x[2] - 25.0, x[1]^2 + x[2]^2 - 25.0]

    jac_g(x) = [x[2] x[1];
    2x[1] 2x[2]]

    xlow = [2.0, 0.0]
    xupp = [50.0, 50.0]

    x0 = 2 * ones(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end