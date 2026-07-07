# Problem 30 from Hock-Schittkowski

function hs30()

    n = 3
    m = 3
    p = 1

    r(x) = [x[1], x[2], x[3]]

    jac_r(x) = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]

    g(x) = x[1]^2 + x[2]^2 - 1.0
    
    jac_g(x) = [2x[1] 2x[2] 0.0]

    xlow = [1.0, -10.0, -10.0]
    xupp = 10 * ones(n)

    x0 = ones(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end