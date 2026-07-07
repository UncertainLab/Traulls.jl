# Problem 23 form Hock-Schittkowski

function hs23()
    
    # Additional slack variable
    n = 3
    m = 2
    p = 4

    r(x) = [x[1], x[2]]
    
    jac_r(x) = [1.0 0.0 0.0; 0.0 1.0 0.0]

    g(x) = [x[1]^2 + x[2]^2 - 1.0,
    9x[1]^2 + x[2]^2 - 9.0,
    x[1]^2 - x[2],
    x[2]^2 - x[1]]

    jac_g(x) = [2x[1] 2x[2] 0.0;
    18x[1] 2x[2] 0.0;
    2x[1] -1.0 0.0;
    -1.0 2x[2] 0.0]

    A = [1.0 1.0 -1.0]
    b = [1.0]

    xlow = [-50.0, -50.0, 0.0]
    xupp = [50.0, 50.0, Inf]

    x0 = [3.0, 1.0, 0.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, A, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end

function hs23_woslack()
    
    n = 2
    m = 2
    p = 5

    r(x) = [x[1], x[2]]
    
    jac_r(x) = [1.0 0.0; 
    0.0 1.0]

    g(x) = [x[1] + x[2] - 1.0,
    x[1]^2 + x[2]^2 - 1.0,
    9x[1]^2 + x[2]^2 - 9.0,
    x[1]^2 - x[2],
    x[2]^2 - x[1]]

    jac_g(x) = [1.0 1.0;
    2x[1] 2x[2];
    18x[1] 2x[2];
    2x[1] -1.0;
    -1.0 2x[2]]

    A = [1.0 1.0 -1.0]
    b = [1.0]

    xlow = [-50.0, -50.0]
    xupp = [50.0, 50.0]

    x0 = [3.0, 1.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end