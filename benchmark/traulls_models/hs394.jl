function hs394()

    n = 20
    m = 2n
    p = 1

    # Residuals 
    function r(x)
        n = size(x, 1)
        rx1 = [sqrt(i)*x[i] for i = 1:n]
        rx2 = [sqrt(i)*x[i]^2 for i = 1:n]

        [rx1; rx2]
    end

    jac_r(x) = ForwardDiff.jacobian(r, x)

    # Constraints 
    c(x) = sum(x.^2) - 1.0

    jac_c(x) = 2 .* x'
    
    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    x0 = 2 * ones(n)

    return Traulls.CnlsModel(r, c, jac_r, jac_c, x_low, x_upp, x0, n, m, p, Val(:only_equalities))
end