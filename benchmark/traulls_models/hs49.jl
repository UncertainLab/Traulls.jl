# Problem 49 from Hock-Schittkowski

function hs49()

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

    h(x) = [x[1] + x[2] + x[3] + 4x[4] - 7.0,
            x[3] + 5x[5] - 6.0]

    jac_h(x) = [1.0 1.0 1.0 4.0 0.0;
                0.0 0.0 1.0 0.0 5.0]

    x_low = fill(-Inf, n)
    x_upp = fill(Inf, n)

    x0 = [10.0, 7.0, 2.0, -3.0, 0.8]

    return Traulls.CnlsModel(r, h, jac_r, jac_h, x_low, x_upp, x0, n, m, p, Val(:only_equalities))

end