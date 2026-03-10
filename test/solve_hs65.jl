@testset "Solving problem 65 form Hock-Shittkowski collection" begin
    
    # Problem 65 form Hock-Schittkowski
    n = 3
    m = 3
    p = 1

    function r!(rx, x)
        rx .= [x[1]-x[2],
               (x[1]+x[2]-10)/3,
               x[3]-5.0]
        return
    end

    function jac_r!(J, x)
        J .= [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]
        return
    end
    # Equality constraints

    function c!(cx, x)
        cx .= [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
        return
    end

    function jac_c!(C, x)
        C .= [-2x[1] -2x[2] -2x[3]]
        return
    end

    # Bounds
    x_low = [-4.5, -4.5, -5.0]
    x_upp = [4.5, 4.5, 5.0]

    # Starting point
    x0 = [-5, 5, 0.0]

    # Testing for model defined with in-place methods
    model = Traulls.BoxCnls!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,Val(:only_inequalities))

    sol = Traulls.solve(model; verbose=true, hessian_approx=Traulls.sr1, output_file_name = "updated_hs65.out")

    @test sol.status isa Traulls.CriticalityStatus
    @test sol.objective isa Real
    @test eltype(sol.primal_vars) <: Real
end
