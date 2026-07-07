# Problem 373 from Hock-Schittkowski

function hs373()

    n = 9
    m = 6
    p = 6

    r(x) = [x[4], x[5], x[6], x[7], x[8], x[9]]

    jac_r(x) = [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

    function c(x)
        rhs = [127, 151, 379, 421, 460, 426]
        cx = [x[1] + x[2] * exp((2i-7) * x[3]) + x[i+3] - rhs[i] for i = 1:6]
        return cx
    end

    jac_c(x) = ForwardDiff.jacobian(c, x)

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x0 = [300.0, -100.0, -0.1997, -127.0, -151.0, 379.0, 421.0, 460.0, 426.0]

    return Traulls.CnlsModel(r, c, jac_r, jac_c, xlow, xupp, x0, n, m, p, Val(:only_equalities))
    
end