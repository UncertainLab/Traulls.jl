# HS353 transformed into feasibility problem 
function hs353()
    n = 5
    m = 1
    p = 1
    # Residuals 
    r(x) = 12x[1] + 11.9x[2] + 41.8x[3] + 52.1x[4] - 
    1.645 * sqrt((0.53x[1])^2 + (0.44x[2])^2 + (4.50x[3])^2 + (0.79x[4])^2) - 12 - x[5]

    jac_r(x) = ForwardDiff.gradient(r, x)'


    g(x) = 2.3x[1] + 5.6x[2] + 11.1x[3] + 1.3x[4] - 5.0

    jac_g(x) = [2.3 5.6 11.1 1.3 0.0]
    
    A = [1.0 1.0 1.0 1.0 0.0]
    b = [1.0]

    xlow = zeros(n)
    xupp = fill(Inf, n)

    x0 = [0.0, 0.0, 0.4, 0.6, 0.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, A, b, xlow, xupp, x0, n, m, p, Val(:only_inequalities))

end