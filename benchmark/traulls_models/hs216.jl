# Problem 216 from Hock-Schittkowski

function hs216()
    n = 2
    m = 2
    p = 1

    r(x) = [10(x[1]^2 - x[2]), x[1] - 1.0]

    jac_r(x) = [20x[1] -10.0;
    1.0 0.0]

    c(x) = x[1] * (x[1] - 4.0) - 2x[2] + 12.0

    jac_c(x) = [2x[1] - 4.0 -2.0]

    xlow = [-Inf, -Inf]
    xupp = [Inf, Inf]

    x0 = [-1.2, 1.0]

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))

end
    