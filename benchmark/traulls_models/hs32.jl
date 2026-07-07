# Problem 32 from Hock-Schittkowski

function hs32()

    n = 3
    m = 2
    p = 1

    r(x) = [x[1] + 3x[2] + x[3], 2(x[1] - x[2])]

    jac_r(x) = [1.0 3.0 1.0;
    2.0 -2.0 0.0]

    g(x) = 6x[2] + 4x[3] - x[1]^3 - 3.0

    jac_g(x) = [-3x[1]^2 6.0 4.0]

    A = [1.0 1.0 1.0]
    b = [1.0]

    xlow = zeros(n)
    xupp = fill(Inf, n)

    x0 = [0.1, 0.7, 0.2]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, A, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end