# Problem 31 form Hock-Schittkowski

function hs31()

    n = 3
    m = 3
    p = 1

    r(x) = [3x[1], x[2], 3x[3]]

    jac_r(x) = [3.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 3.0]

    g(x) = x[1] * x[2]- 1.0
    
    jac_g(x) = [x[2] x[1] 0.0]

    xlow = [-10.0, 1.0, -10.0]
    xupp = [10.0, 10.0, 1.0]

    x0 = ones(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end