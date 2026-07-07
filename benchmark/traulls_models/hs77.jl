# Problem 77 form Hock-Schittkowski

function hs77()

    n = 5
    m = 5
    p = 2

    r(x) = [x[1] - 1.0, 
    x[1] - x[2], 
    x[3] - 1.0,
    (x[4] - 1.0)^2,
    (x[5] - 1.0)^3]

    jac_r(x) = [1.0 0.0 0.0 0.0 0.0;
    1.0 -1.0 0.0 0.0 0.0;
    0.0 0.0 1.0 0.0 0.0;
    0.0 0.0 0.0 2(x[4]-1.0) 0.0;
    0.0 0.0 0.0 0.0 3(x[5]-1.0)^2]

    c(x) = [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2),
    x[2] + x[3]^4 * x[4]^2 - 8.0 - sqrt(2)]

    jac_c(x) = [2x[1]*x[4] 0.0 0.0 x[1]^2+cos(x[4]-x[5]) -cos(x[4]-x[5]);
    0.0 1.0 4x[3]^3*x[4]^2 2x[4]*x[3]^4 0.0]

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = 2 * ones(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
    
end