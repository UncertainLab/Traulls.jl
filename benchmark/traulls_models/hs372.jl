# Problem 372 from Hock-Schittkowski

function hs372()

    n = 9
    m = 6
    p = 12

    r(x) = [x[4], x[5], x[6], x[7], x[8], x[9]]

    jac_r(x) = [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

    function g(x)
        
        rhs = [127, 151, 379, 421, 460, 426]

        gx1 = [x[1] + x[2] * exp((2i-7) * x[3]) + x[i+3] - rhs[i] for i = 1:6]
        gx2 = [-x[1] - x[2] * exp((2i-7) * x[3]) + x[i+3] + rhs[i] for i = 1:6]

        return vcat(gx1, gx2)
    end

    jac_g(x) = ForwardDiff.jacobian(g, x)

    xlow = [-Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    xupp = fill(Inf, n)

    x0 = [300.0, -100.0, -0.1997, -127.0, -151.0, 379.0, 421.0, 460.0, 426.0]

    return Traulls.CnlsModel(r, g, jac_r, jac_g, xlow, xupp, x0, n, m, p, Val(:only_inequalities))
    
end