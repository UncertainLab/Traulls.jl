# [Traulls.jl documentation](@id Home)

## Introduction 

`Traulls.jl` package implements a solver for constrained nonlinear least-squares problems of the form

```math
\begin{aligned}
\min_{x} \quad &  \dfrac{1}{2} \|r(x)\|^2 \\
\text{s.t.} \quad & c(x) = 0\\
& Ax = b \\
& \ell \le x \le u,
\end{aligned}
```

where the residuals  $r \colon \mathbb{R}^n \to \mathbb{R^{n_r}}$ and the constraints $c \colon \mathbb{R}^n \to \mathbb{R}^{n_c}$ are two time continuously differentiable, the $m \times n$ equality constraint matrix $A$ (with $m < n$) is full row rank, vectors $\ell, u$ are bounds 
on the variables with components in $\mathbb{R} \cup \pm \infty$, and $\|\cdot\|$ denotes the euclidean norm. 

## Installation

To install the package from the Github repository, open the Julia REPL and type the following command:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/UncertainLab/Traulls.jl")
```

## Basic usage



## Bug reports and contributions

If you encountered a bug while using the package, you can open an [issue](https://github.com/UncertainLab/Traulls.jl/issues) to report it. Issues can also be opened to discuss about suggestions of improvements.