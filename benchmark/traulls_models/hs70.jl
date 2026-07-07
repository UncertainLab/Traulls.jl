# Problem 70 from Hock-Schittkowski

function hs70()

    n = 4
    m = 19
    p = 1

    # Data
    c = [(i >= 2) ? i - 1 : 0.1 for i = 1:19]
    yobs = [
    0.00189,
    0.1038,
    0.268,
    0.506,
    0.577,
    0.604,
    0.725,
    0.898,
    0.947,
    0.845,
    0.702,
    0.528,
    0.385,
    0.257,
    0.159,
    0.0869,
    0.0453,
    0.01509,
    0.00189]
    
    ycal(x, c) = begin b = x[3] + (1.0 - x[3]) * x[4]
        [(1 + 1 / (12 * x[2])) * (x[3] * b^x[2]) * ((x[2] / 6.2832)^(0.5)) * 
    (c[i] / 7.685)^(x[2] - 1) * exp(x[2] - b * c[i] * x[2] / 7.658) + 
    (1 + (1 / (12 * x[1]))) * (1 - x[3]) * (b / x[4])^x[1] * (x[1] / 6.2832)^0.5 *
    (c[i] / 7.658)^(x[1] - 1) * exp(x[1] - b * c[i] * x[1] / (7.658 * x[4])) for i = 1:19] end

    r(x) = ycal(x, c) .- yobs

    jac_r(x) = ForwardDiff.jacobian(r, x)

    g(x) = x[3] + (1 - x[3]) * x[4] 
     
    jac_g(x) = [0.0 0.0 1.0-x[4] 1.0-x[3]]

    xlow = [1e-5, 1e-5, 1e-5, 1e-5]
    xupp = [100.0, 100.0, 1.0, 100.0]

    x0 = [2.0, 4.0, 0.04, 2.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
end