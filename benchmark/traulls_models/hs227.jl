# Problem 227 from Hock-Schittkowski

function hs227()
    n = 2
    m = 2
    p = 2

    r(x) = [x[1] - 2.0, x[2] - 1.0]
    
    jac_r(x) = [1.0 0.0;
                0.0 1.0]

    g(x) = [-x[1]^2 + x[2],
    x[1] - x[2]^2]
    
    jac_g(x) = [-2x[1] 1.0;
    1.0 -2x[2]]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = 1/2 * ones(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end