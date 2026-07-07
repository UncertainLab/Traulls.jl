# Problem 57 from Hock & Schittkowski collection

function hs57()

    # Dimensions
    n = 2
    m = 44
    p = 1

    # Data

    a = [8.,8.,10.,10.,10.,10.,12.,12.,12.,12.,14.,14.,14.,16.,16.,16.,18.,18.,20.,20.,20.,22.,22.,22.,
         24.,24.,24.,26.,26.,26.,28.,28.,30.,30.,30.,32.,32.,34.,36.,36.,38.,38.,40.,42.]

    b = [.49,.49,.48,.47,.48,.47,.46,.46,.45,.43,.45,.43,.43,.44,.43,.43,.46,.45,.42,.42,.43,.41,
         .41,.40,.42,.40,.40,.41,.40,.41,.41,.40,.40,.40,.38,.41,.40,.40,.41,.38,.40,.40,.39,.39]

    # Residuals

    f(x,input,output) = output .- (t -> x[1] + (0.49 - x[1]) * exp(-x[2]*(t - 8))).(input)
    r(x) = f(x,a,b)

    jac_r(x) = ForwardDiff.jacobian(r,x)

    # Constraints

    c(x) = [0.49*x[2]-x[1]*x[2]-0.09]
    jac_c(x) = [-x[2] 0.49-x[1]]

    x_low = [0.4, -4]
    x_upp = fill(Inf,n)

    # Starting point
    x0 =  [0.42,5.0]
    x = vcat(x0,c(x0))

    return Traulls.CnlsModel(r, c, jac_r, jac_c, x_low, x_upp,x0, n, m, p, Val(:only_inequalities))
end
