# Problem 264 from Hock-Schittkowski

function hs264()

    n = 4
    m = 4
    p = 3

    r(x) = [x[1] - 5 / 2, x[2] - 5 / 2, sqrt(2) * (x[3] - 21 / 4), x[4] + 7 / 2]

    jac_r(x) = [1.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0;
    0.0 0.0 sqrt(2) 0.0;
    0.0 0.0 0.0 1.0]

    g(x) = [-x[1]^2 - x[2]^2 - x[3]^2 - x[4]^2 - x[1] + x[2] + x[3] + x[4] + 8.0,
    -x[1]^2 - 2 * x[2]^2 - x[3]^2 - 2 * x[4]^2 + x[1] + x[4] + 9.0,
    -2 * x[1]^2 - x[2]^2 - x[3]^2 - 2 * x[1] + x[2] + x[4] + 5.0]

    jac_g(x) = [-2x[1]-1.0 -2x[2]+1.0 -2x[3]+1.0 -2x[4]+1.0;
    -2x[1]+1.0 -4x[2] -2x[3] -4x[4]+1.0;
    -4x[1]-2.0 -2x[2]+1.0 -2x[3] 1.0]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = zeros(n)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end