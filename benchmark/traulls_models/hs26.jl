# Problem 26 from Hock-Schittkowski

function hs26()

    n = 3
    m = 2
    p = 1

    r(x) = [x[1] - x[2], (x[2] - x[3])^2]

    jac_r(x) = [1.0 -1.0 0.0;
    0.0 2(x[2]-x[3]) -2(x[2]-x[3])]

    c(x) = (1 + x[2]^2) * x[1] + x[3]^4 - 3.0

    jac_c(x) = [1+x[2]^2 2x[1]*x[2] 4x[3]^3]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = [-2.6, 2.0, 2.0]

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
end