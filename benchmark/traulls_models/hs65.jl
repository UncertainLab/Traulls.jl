# Problem 65 from Hock & Schittkowski collection


function hs65()

    # Dimensions

    n = 3
    m = 3
    p = 1

    # Residuals
    r(x) = [x[1]-x[2],
            (x[1]+x[2]-10)/3,
            x[3]-5.0]

    jac_r(x) = [1. -1. 0;
                        1/3 1/3 0.;
                        0. 0. 1.;]

    # Equality constraints
    c(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
    jac_c(x) = [-2x[1] -2x[2] -2x[3]]

    # Bounds
    x_low = [-4.5, -4.5, -5.0]
    x_upp = [4.5, 4.5, 5.0]

    # Starting point
    x0 = [-5, 5, 0.0]

    return Traulls.CnlsModel(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,Val(:only_inequalities))
end
