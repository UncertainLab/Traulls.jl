# Problem 323 from Hock-Schittkowski

function hs323()
    
    # Additional slack variable for linear equality constraints
    n = 3
    m = 2
    p = 1

    r(x) = [x[1] - 2.0, x[2]]

    jac_r(x) = [1.0 0.0 0.0;
    0.0 1.0 0.0]

    g(x) = -x[1]^2 + x[2] - 1.0

    jac_g(x) = [-2x[1] 1.0 0.0]

    A = [1.0 -1.0]
    b = [-2.0]
    eqmat = hcat(A, [-1.0])
    xlow = zeros(n)
    xupp = fill(Inf, n)

    xstart = [0, 1]
    x0 = vcat(xstart, A*xstart-b)

    return Traulls.CnlsModel(r, g, jac_r, jac_g, eqmat, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end