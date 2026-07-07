# Problem 6 from Hock Schittkowski

function hs6()

    n = 2
    m = 1
    p = 1

    r(x) = [1.0 - x[1]]
    jac_r(x) = [-1.0 0.0]

    c(x) = 10.0 * (x[2] - x[1]^2)
    jac_c(x) = [-20x[1] 10.0]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = [-1.2, 1.0]

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
end