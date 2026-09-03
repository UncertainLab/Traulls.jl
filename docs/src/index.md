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

where the residuals  $r$ and the constraints $c$ are two time continuously differentiable, equality constraint matrix $A$ is full row rank and $\ell, u$ are bounds 
on the variables. 