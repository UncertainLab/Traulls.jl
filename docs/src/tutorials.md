# [Tutorials](@id Tutorials)


```@meta
CurrentModule = Traulls
```

```@setup tutorial
using Traulls
```

This section provides details on how to use `Traulls.jl` to solve a constrained nonlinear least-squares problems of the form

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


## Instantiating a model

The package provides a basic interface to instantiate a model of the form 

```@docs; canonical = false
CnlsModel{T}
```

## Solving

Once a `CnlsModel` has been instantiated, you can call the function the main solving function `traulls` on it.

```@docs; canonical = false
traulls
```

## Execution info

```@docs; canonical = false
Traulls.TraullsResults
```

## Examples