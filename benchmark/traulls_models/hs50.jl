# Problem 50 from Hock-Schittkowski

function hs50()

    n = 5
    m = 4
    p = 3

    r(x) = [x[1] - x[2],
            x[2] - x[3],
            (x[3] - x[4])^2,
            x[4] - x[5]]

    jac_r(x) = [1.0 -1.0 0.0 0.0 0.0;
                0.0 1.0 -1.0 0.0 0.0;
                0.0 0.0 2*(x[3] - x[4]) 2*(x[4]-x[3]) 0.0;
                0.0 0.0 0.0 1.0 -1.0]

    h(x) = [x[1] + 2x[2] + 3x[3] - 6.0,
            x[2] + 2x[3] + 3x[4] - 6.0,
            x[3] + 2x[4] + 3x[5] - 6.0]

    jac_h(x) = [1.0 2.0 3.0 0.0 0.0;
                0.0 1.0 2.0 3.0 0.0;
                0.0 0.0 1.0 2.0 3.0]

    x_low = fill(-Inf, n)
    x_upp = fill(Inf, n)

    x0 = [-35.0, -31.0, 11.0, 5.0, -5.0]

    return Traulls.CnlsModel(r, h, jac_r, jac_h, x_low, x_upp, x0, n, m, p, Val(:only_equalities))

end