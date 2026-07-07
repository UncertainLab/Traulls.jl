# Problem 79 from Hock-Schittkowski

function hs79()

    n = 5
    m = 5
    p = 3

    r(x) = [x[1] - 1.0, 
    x[1] - x[2], 
    x[2] - x[3],
    (x[3] - x[4])^2,
    (x[4] - x[5])^2]

    jac_r(x) = [1.0 0.0 0.0 0.0 0.0;
    1.0 -1.0 0.0 0.0 0.0;
    0.0 1.0 -1.0 0.0 0.0;
    0.0 0.0 2(x[3] - x[4]) -2(x[3] - x[4]) 0.0;
    0.0 0.0 0.0 2(x[4] - x[5]) -2(x[4] - x[5])]

    c(x) = [x[1] + x[2]^2 + x[3]^3 - 2.0 - 3 * sqrt(2),
    x[2] - x[3]^2 + x[4] + 2.0 - 2.0 * sqrt(2),
    x[1] * x[5] - 2.0]

    jac_c(x) = [1.0 2x[2] 3x[3]^2 0.0 0.0;
    0.0 1.0 -2x[3] 1.0 0.0;
    x[5] 0.0 0.0 0.0 x[1]]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = 2 * ones(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
    
end