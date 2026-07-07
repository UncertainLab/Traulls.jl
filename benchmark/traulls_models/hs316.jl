# Problem 316 from Hock-Schittkowski

function hs316()

    n = 2
    m = 2
    p = 1

    r(x) = [x[1] - 20.0, x[2] + 20.0]

    jac_r(x) = [1.0 0.0
    0.0 1.0]

    c(x) = x[1]^2 * (1/100) + x[2]^2 * (1/100) - 1.0

    jac_c(x) = [x[1]*(1/50) x[2]*(1/50)]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = zeros(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
end