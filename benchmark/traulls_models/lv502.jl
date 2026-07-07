# Problem 5.2 from Luksan & Vlsek collection (1999)
# Chained Wood function with Broyden banded constraints

# Dimensions


function lv502(n::Int=100)

if n < 8
    @warn "n must be greater or equal than 8, set to 8"
    n = 8
end

if n % 2 != 0
    @warn "n must be even, rounding up"
    n += 1
end

N = div(n,2) - 1
m = 6N
p = n-7

# Residuals
function r!(rx, x)
    N = div(size(x,1),2) - 1
    s = sqrt(10)

    for i = 1:N
        rx[i] = 10(x[2i-1]^2 - x[2i])
        rx[N+i] = x[2i-1] - 1
        rx[2N+i] = 3s*(x[2i+1]^2 - x[2i+2])
        rx[3N+i] = x[2i+1]-1
        rx[4N+i] = s*(x[2i] + x[2i+2] - 2)
        rx[5N+i] = (x[2i] - x[2i+2])*(1/s)
    end
    return
end

jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

# Sparse version of the jacobian
function jr!(J, x)
    n = size(x, 1)
    N = div(n, 2) - 1
    sqrtten = sqrt(10)

    for i = 1:N
        J[i, 2i-1] = 20*x[2i-1]
        J[i, 2i] = -10
        J[N+i, 2i-1] = 1
        J[2N+i, 2i+1] = 6*sqrtten*x[2i+1]
        J[2N+i, 2i+2] = -3*sqrtten
        J[3N+i, 2i+1] = 1
        J[4N+i, 2i] = sqrtten
        J[4N+i, 2i+2] = sqrtten
        J[5N+i, 2i] = 1/sqrtten
        J[5N+i, 2i+2] = -1/sqrtten
    end
    return
end

# Sparsity pattern
jr_nzrows = Vector{Int}([])
jr_nzcols = Vector{Int}([])

for i=1:N
    append!(jr_nzrows, [i, i]); append!(jr_nzcols, [2i-1], 2i)
    append!(jr_nzrows, [N+i]); append!(jr_nzcols, [2i-1])
    append!(jr_nzrows, [2N+i, 2N+i]); append!(jr_nzcols, [2i+1, 2i+2])
    append!(jr_nzrows, [3N+i]); append!(jr_nzcols, [2i+1])
    append!(jr_nzrows, [4N+i, 4N+i]); append!(jr_nzcols, [2i, 2i+2])
    append!(jr_nzrows, [5N+i, 5N+i]); append!(jr_nzcols, [2i, 2i+2])
end

# Constraints
function c!(cx, x)
    n = size(x,1)
    cx .= [(2+5x[k+5]^2)*x[k+5] + 1 + sum(x[i]*(1+x[i]) for i=max(k-5,1):k+1) for k=1:n-7]
    return
end

jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(p), x)

# Sparse version of the Jacobian
function jc!(C, x)
    n = size(x, 1)

    for k=1:n-7
        C[k, k+5] = 2 + 15 * x[k+5]^2

        for i = max(k-5, 1) : k+1
            C[k, i] = 1 + 2x[i]
        end
    end
    return
end

# Sparsity pattern
jc_nzrows = Vector{Int}([])
jc_nzcols = Vector{Int}([])

for k=1:n-7
    push!(jc_nzrows, k); push!(jc_nzcols, k+5)
    for i = max(k-5, 1):k+1
        push!(jc_nzrows, k); push!(jc_nzcols, i)
    end
end

x_low = fill(-Inf, n)
x_upp = fill(Inf, n)

# Starting point
x = [(mod(i,2) == 1 ? -2. : 1.) for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))

return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
