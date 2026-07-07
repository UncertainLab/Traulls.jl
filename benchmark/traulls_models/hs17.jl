
# Problem 17 from Hock-Schittkowski

function hs17()

    n = 2
    m = 2
    p = 2

    r(x) = [10(x[2] - x[1]^2), 1.0-x[1]]

    jac_r(x) = [-20x[1] 10.0;
    -1.0 0.0]

    g(x) = [x[2]^2 - x[1], x[1]^2 - x[2]]

    jac_g(x) = [-1.0 2x[2];
    2x[1] -1.0]

    xlow = [-0.5, -Inf]
    xupp = [0.5, 1.0]

    x0 = [-2.0, 1.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
    
end