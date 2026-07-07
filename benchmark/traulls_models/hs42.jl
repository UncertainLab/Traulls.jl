# Problem 42 from Hock & Schittkowski collection

function hs42()

    # Dimensions
    n = 4
    m = 4
    p = 2

    # Residuals
    r(x) = [x[1]-1.0,
            x[2]-2.0,
            x[3]-3.0,
            x[4]-4.0]

    jac_r(x) = [1. 0 0 0;
                0 1 0 0;
                0 0 1 0;
                0 0 0 1]

    # Equality constraints
    c(x) = [x[1] - 2.0, x[3]^2 + x[4]^2 - 2]
    jac_c(x) = [1.0 0 0 0;
                0 0 2*x[3] 2*x[4]]

    # Bounds
    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    x0 = ones(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, x_low, x_upp, x0, n, m, p, Val(:only_equalities))
end
