function hs354()
    n = 4
    m = 4
    p = 1

    # Residuals
    r(x) = [x[1] + 10x[2],
    sqrt(5) * (x[3] - x[4]),
    (x[2] - 2x[3])^2,
    sqrt(10) * (x[1] - x[4])^2]

    jac_r(x) = [1.0 10.0 0.0 0.0;
    0.0 0.0 sqrt(5) -sqrt(5);
    0.0 2*(x[2]-2x[3]) -4*(x[2]-2x[3]) 0.0
    2*sqrt(10)*(x[1]-x[4]) 0.0 0.0 -2*sqrt(10)*(x[1]-x[4])]

    # Constraints
    g(x) = x[1] + x[2] + x[3] +x[4] - 1.0

    jac_g(x) = [1.0 1.0 1.0 1.0]

    xlow = fill(-Inf, n)
    xupp = 20*ones(n)

    x0 = [3.0, -1, 0, 1]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end