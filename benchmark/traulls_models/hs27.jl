# Problem 27 from Hock-Schittkowski

function hs27()

    n = 3
    m = 2
    p = 1

    r(x) = [0.1 * (x[1] - 1), x[2] - x[1]^2]

    jac_r(x) = [0.1 0.0 0.0;
    -2x[1] 1.0 0.0]

    c(x) = x[1] + x[3]^2 + 1.0

    jac_c(x) = [1.0 0.0 2x[3]]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)
    
    x0 = 2 * ones(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
end