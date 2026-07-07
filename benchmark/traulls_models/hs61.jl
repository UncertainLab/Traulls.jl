# Problem 61 from Hock-Schittkowski

function hs61()

    n = 3
    m = 3
    p = 2

    r(x) = [2 * (x[1] - 33 / 8), sqrt(2) * (x[2] + 4), sqrt(2) * (x[3] - 6)]

    jac_r(x) = [2.0 0.0 0.0;
    0.0 sqrt(2) 0.0;
    0.0 0.0 sqrt(2)]

    c(x) = [3x[1] - 2x[2]^2 - 7.0,
    4x[1] - x[3]^2 - 11.0]

    jac_c(x) = [3.0 -4x[2] 0.0;
    4.0 0.0 -2x[3]]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = zeros(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
end