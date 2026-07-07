# Problem 13 from Hock-Schittkowski

function hs13()
    n = 2
    m = 2
    p = 1

    r(x) = [x[1] - 2.0, x[2]]
    
    jac_r(x) = [1.0 0.0;
                0.0 1.0]

    g(x) = (1 - x[1])^3 - x[2]
    
    jac_g(x) = [-3*(1 - x[1])^2 -1.0]

    xlow = zeros(n)
    xupp = fill(Inf, n)

    x0 = -2 * ones(2)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end