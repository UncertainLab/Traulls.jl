# Problem 46 from Hock-Schittkowski

function hs46()

    n = 5
    m = 4
    p = 2

    r(x) = [x[1] - x[2],
            x[3] - 1.0,
            (x[4] - 1.0)^2,
            (x[5] - 1.0)^3]

    jac_r(x) = [1.0 -1.0 0.0 0.0 0.0;
                0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 2*(x[4]-1.0) 0.0;
                0.0 0.0 0.0 0.0 3*(x[5]-1.0)^2]

    h(x) = [x[1]^2 * x[4] + sin(x[4] - x[5]) - 1.0,
            x[2] + x[3]^4 * x[4]^2 - 2.0]

    jac_h(x) = [2*x[1]*x[4] 0.0 0.0 (x[1]^2 + cos(x[4]-x[5])) -cos(x[4]-x[5]);
                0.0 1.0 4*x[3]^3*x[4]^2 2*x[3]^4*x[4] 0.0]

    x_low = fill(-Inf, n)
    x_upp = fill(Inf, n)

    x0 = [sqrt(2) / 2, 1.75, 0.5, 2.0, 2.0]

    return Traulls.CnlsModel(r, h, jac_r, jac_h, x_low, x_upp, x0, n, m, p, Val(:only_equalities))

end