"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation 
- `sr1`: second-order terms updates by a SR1 formula
"""

@enum HessianApprox gn sr1

"""
    GN <: ALHessian 

Mutable structure representing the Gauss-Newton approximation of the Augmented Lagrangian Hessian 

**Attributes**

* `J`: Jacobian of the residuals 

* `C`: Jacobian of the nonlinear constraints 

* `μ`: penalty parameter

The resulting approximation is `H = JᵀJ + μCᵀC`.

See [`Base.:*(H::GN, v)`](@ref), [`vthv(H::GN, v)`](@ref)
"""
mutable struct GN <: ALHessian
    J::Matrix
    C::Matrix
    mu::Float64 
end

""" vthv(H::GN,v::Vector)

The quadratic term `vᵀHv` where `H = JᵀJ + μCᵀC` is the Gauss-Newton approximation of the augmented Lagrangian Hessian encoded into
the [`GN`](@ref) type.
"""
function vthv(H::GN, v::Vector) 
    Jv = H.J*v
    Cv = H.C*v 
    return dot(Jv,Jv) + H.mu*dot(Cv,Cv)
end 

""" Base.:*(H::GN, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid matrix-matrix multiplication
"""
function Base.:*(H::GN, v::Vector) 
    return H.J' * (H.J*v) + H.C' * (H.mu*H.C*v)
end

"""
    update_hessian!(H,J₊,C₊)

Updates the Gauss-Newton Hessian approximation `H` by modifiying the `J` and `C` attributes with, 
respectively `J₊` and `C₊`.
"""
function update_hessian!(
    H::GN,
    J_new::Matrix,
    C_new::Matrix) 
    
    H.J = J_new
    H.C = C_new
    return
end

"""
    HybridSR1 <: ALHessian

Mutable structure reprensenting the SR1 approximation of the Augmented Lagrangian Hessian

**Attributes**

* `J`: Jacobian of the residuals

* `C`: Jacobian of the nonlinear constraints

* `S`: Approximation of the second order terms of the true Hessian

* `mu`: penalty parameter 

* `small_res`: Boolean indicating if the second order terms `S` can be neglected in computations

* `secant_rhs`: Right handside of the structured secant equation the SR1 is based on

"""
mutable struct HybridSR1 <: ALHessian
    J::Matrix
    C::Matrix
    S::Matrix
    mu::Float64
    small_res::Bool
    secant_rhs::Vector
end

"""
    HybridSR1(J,C,μ)

Constructor for the [`HybridSR1`](@ref) structure. 

One can initlize a `HybridSR1` with given Jacobians `J`, `C` and penalty parameter `mu`.

The remaining attributes are initialized as follows.

* `S`: zero matrix of appropriate dimensions

* `small_res`: `true`

* `secant_rhs`: zero vector of appropriate dimension

This choice of initialization is tantamount to encode the Gauss-Newton approximation into a `HybridSR1`. 
"""
function HybridSR1(
    J::Matrix,
    C::Matrix,
    mu::Float64)

    n = size(J,2)

    return HybridSR1(J,C,zeros(n,n),mu,true,zeros(n))
end

""" Base.:*(H::HybridSR1,v::Vector)

Overloads the `*` operator to the type [`HybridSR1`](@ref) for better Hessian-vector product.
Avoids to perform matrix-matrix multiplication. 
"""
function Base.:*(H::HybridSR1, v::Vector) 
    
    Hv = H.J' * (H.J*v) .+ H.mu*H.C' * (H.C*v)
    
    if !H.small_res Hv .+= H.S*v end 

    return Hv
end

"""
     update_hessian!(H,s,φ,φ₊,κ_sos,κ_smlres)

Updates the `HybridSR1` Hessian `H` by applying the SR1 formula on the second order terms attribute `S`.

After the update, a criterion is evaluated to assert wether or not the second order terms can be neglected.

**Arguments**

- `H`: current Hessian approximation
- `s`: step for the current iteration
- `φ`: value of the Augmented Lagrangian at current iterate
- `φ₊`: value of the Augmented Lagrangian at next iterate
- `κ_sos`: small positive constant used in the safeguard against small denominator
- `κ_smlres`: small positive constant used to check if the second order terms can be neglected or not

**On return**

Modifies in place  the attributes `S` and `small_res` of `H`. 
"""
function update_hessian!(
    H::HybridSR1,
    s::Vector,
    mx::Float64,
    mx_next::Float64,
    kappa_sos::Float64,
    kappa_sml_res::Float64)

    # Update matrix fields of the Hessian
    update_sr1_second_order(H,s,kappa_sos)

    # In current version, Jacobians are implicitly updated when evaluated at trial point 
    # TODO: think about a more modular way of handeling Jacobians in place modifications
    # H.J .= J_next[:,:]
    # H.C .= C_next[:,:]

    # Update small_residuals field
    # check_small_residuals(H,mx,mx_next,kappa_sml_res)
    H.small_res = false

    return
end

"""
    update_sr1_second_order!(H,s,κ_sos)

Update the second order terms of the Hessian approximation `H` with a SR1 formula based on
the structured secant equation `Sᵀs = yₐ` where `yₐ` denotes the `secant_rhs` attribute of `H`.

First, a safeguard is tested to check if the denominator of the update formula is too small.

The update is skipped if `Sᵀs = yₐ` or `|(Sᵀs-yₐ)ᵀs| < κ_sos *||s||*||Sᵀs-yₐ||`.

If not, we add `(Sᵀs-yₐ)(Sᵀs-yₐ)ᵀ / (sᵀ(Sᵀs-yₐ))` to the current second order approximation

**Arguments**

- `H`: Structure of type [`HybridSR1`](@ref) encoding the current Hessian approximation
- `s`: step of the current iteration
- `κ_sos`: small positive constant used in the small denominator safeguard

**On return**

Modifies in place the attribute `S` of `H`.
"""
function update_sr1_second_order(
    H::HybridSR1,
    s::Vector,
    kappa_sos::Float64)

    atol = sqrt(eps(Float64))

    v = H.secant_rhs .- H.S*s
    stv = dot(s,v)
    norm_s = norm(s)
    norm_v = norm(v)

    # Safeguard: update skipped if the denominator `sᵀv` is too small
    skip_update = norm(v,Inf) < atol || abs(stv) < kappa_sos * norm_s * norm_v
    
    if !skip_update H.S .+= (1/stv) .* v*v' end
    
    return
end

"""
    check_small_residuals(H,φ,φ₊,κ_smlres)

Tests if the relative reduction `(φ - φ₊) / φ` is below the treshold value `κ_smlres`.
This heuristic estimates if the residuals are likely to be small at the solution,
and hence if one can neglect the second order terms of the Hessian approximation.

Sets `small_res` attribute of `H` to the boolean result of  `(φ - φ₊) / φ < κ_smlres`.

**Arguments**

- `H`: structure of type [`HybridSR1`](@ref) encoding the current Hessian approximation
- `φ`: value of the objective function at current iterate
- `φ₊`: value of the objective function at next iterate
- `κ_smlres`: treshold value of the heuristic test

**On return**

Modifies in place the `small_res` attribute of `H`.
"""
function check_small_residuals(
    H::HybridSR1,
    mx::Float64,
    mx_next::Float64,
    kappa_sml_res::Float64)

    H.small_res = (mx - mx_next) < kappa_sml_res * mx
    return  
end

"""
    AlSR1Hessian <: ALHessian 

Mutable structure representing the strutured SR1 approximation of the Augmented Lagrangian Hessian.

**Attributes**

* `J`: Jacobian of the residuals 

* `C`: Jacobian of the nonlinear constraints 

* `S`: approximation of the second order terms in the true Hessian

* `μ`: penalty parameter  

The resulting approximation is `H = JᵀJ + μCᵀC + S`.

In order to not explicitely form the matrix-matrix terms, frequent operations are overloaded and take advantage of the Hessian structure. 

If `v` is a vector, this includes the computation of the matrix-vector product `Hv` or the quadratic term `vᵀHv`.

See [`Base.:*(H::AlSR1Hessian, v)`](@ref), [`vthv(H::AlSR1Hessian, v)`](@ref)
"""
mutable struct AlSR1Hessian <: ALHessian
    J::Matrix
    C::Matrix
    S::Matrix
    mu::Float64 
end

""" vthv(H::AlSR1Hessian,v)

The quadratic term `vᵀHv` where `H = JᵀJ + μCᵀC + S` is an SR1 approximation of the augmented Lagrangian Hessian encoded into
the [`AlSR1Hessian`](@ref) type.
"""
function vthv(H::AlSR1Hessian, v::Vector) 
    Jv = H.J*v
    Cv = H.C*v 
    return dot(Jv,Jv) + H.mu*dot(Cv,Cv) + dot(v,H.S*v)
end 



""" Base.:*(H::AlSr1Hessian, v)

Overload the `*` operator to the type [`SR1Hessian`](@ref) in order to avoid matrix-matrix multiplication 
"""
function Base.:*(H::AlSR1Hessian, v::Vector) 
    Jv = H.J*v 
    muCv = H.mu*H.C*v 
    return H.J' * Jv + H.C' * muCv + H.S*v
end

# Compute the approximated 2nd order terms of Hessian with SR1 method  

"""
    sr1_update

Update the SR1 approximation of the second order terms of the Augmented Lagrangian Hessian.

Uses a safeguard to avoid the small denominator degenerate cases.

Returns the updated matrix.
"""
function sr1_update(
    s::Vector,
    g_next::Vector,
    r_next::Vector,
    y_bar_next::Vector,
    J::Matrix,
    C::Matrix,
    S::Matrix,
    kappa_sfg::Float64) 

    y_a = g_next - J'*r_next - C'*y_bar_next
    v = y_a - S*s
    stv = dot(s,v)
    sfg_treshold = kappa_sfg*dot(v,v) 
    S_next = deepcopy(S)

  
    if stv >= sfg_treshold
        S_next .+= (1/stv) .* (v*v')
    end
    return S_next
end

"""
    hybrid_update_hessian

Updates the approximation of the Augmented Lagrangian Hessian using a hybrid strategy.

First, the SR1 update is computed and a heuristic test is then used to detect if the residuals are small or not, and so if the second order terms can be neglected.

For small residuals, the Hessian is updated with the Gauss-Newton approximation.

In the other case, the Hessian is updated with the SR1 approximation.

**On return**

* `H`: Updated approximation of the Hessian 

* `S_next`: Approximated second order terms with updated with the SR1 update formula
"""
function update_hybrid_hessian(
    s::Vector,
    mx::Float64,
    mx_next::Float64,
    g_next::Vector,
    r_next::Vector,
    J_next::Matrix,
    C_next::Matrix,
    mu::Float64,
    y_bar_next::Vector,
    J::Matrix,
    C::Matrix,
    S::Matrix,
    kappa_smlres::Float64,
    kappa_sfg::Float64)  

    S_next = sr1_update(s,g_next,r_next,y_bar_next,J,C,S,kappa_sfg)
    small_residuals = (mx-mx_next) / mx < kappa_smlres
    H = (small_residuals ? GN(J_next,C,mu) : AlSR1Hessian(J_next,C_next,S_next,mu))
    
    return H, S_next
end

"""
    update_hessian_w_sr1

Updates the approximation of the Hessian of the Augmented Lagrangian with the SR1 approach.

**On return**

* `H`: Updated approximation of the Hessian 

* `S_next`: Approximated second order terms with updated with the SR1 update formula
"""
function update_hessian_w_sr1(
    s::Vector,
    g_next::Vector,
    r_next::Vector,
    J_next::Matrix,
    C_next::Matrix,
    mu::Float64,
    y_bar_next::Vector,
    J::Matrix,
    C::Matrix,
    S::Matrix,
    kappa_sfg::Float64) 


    verbose = false
    S_next = sr1_update(s,g_next,r_next,y_bar_next,J,C,S,kappa_sfg)
    verbose && println("[update_hessian]")
    verbose && @show S_next 
    H = AlSR1Hessian(J_next,C_next,S_next,mu)
    return H, S_next
end
