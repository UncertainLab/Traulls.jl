# Global convergence of the inner iterations with the exact projected gradient path

*Traulls.jl, COAP revision. Companion to `cauchy-projected-gradient-path.md`, whose open
points 1 and 2 (§9) it settles. Written 2026-10-24.*

This note proves the global convergence of the inner trust-region iterations (Algorithm 2 of
the paper) when the Cauchy step follows the exact projected gradient path over the polyhedron
$\Omega\cap\{\|s\|_\infty\le\Delta\}$. It works in an idealized setting: exact arithmetic,
exact path, no tolerances. The parking, tie and noise tolerances of the implementation are
not discussed.

---

## 0. Summary

1. **The first local minimizer alone has no uniform decrease guarantee over a polyhedron.**
   The model decrease it provides cannot be bounded below by any positive function of
   $h(x)=\|P_\Omega(x-g)-x\|$, $\pi_\varepsilon(x)$, $\Delta$ and $\|H\|$. This is not a
   weakness of the proof. A four-variable example (§3.2) has $\Delta=1$, $H\succeq0$ with
   $\|H\|=2$, $h\to\sqrt2/10$ and $\pi_\varepsilon=0.2$, while the decrease at the first local
   minimizer is $3.7\delta+O(\delta^2)\to0$. It is the zigzag example of the paper plus one
   free variable, which turns the stationary piece into a *slow* piece.
2. **The cause is the releases.** Along a piece, the speed $\|s'(t)\|$ is the norm of the
   reduced gradient. It decreases at a hit and increases at a release (Lemma 2). If no bound
   is released after the first local minimizer, as on a box, the former argument is correct
   and gives the decrease of §3.1 with no safeguard (Proposition 4). The old draft's error
   (nested tangent spaces) is exactly the missing release case.
3. **A one-parameter safeguard restores the guarantee.** Let $L(t)=-\langle g,s(t)\rangle$
   and let $t_G$ be the first $t>0$ at which $\psi(t)=q(s(t))$ reaches $-\kappa_G L(t)$, with
   $\kappa_G\in(0,1)$. The first local minimizer is kept whenever $t_G$ (or the end of the
   path) comes first. Otherwise the scan goes on to the next local minimizer after $t_G$ and
   returns the best one met (§4.1). In random tests the step changes by less than 0.7%, and
   the safeguard costs one extra piece in the median case (§7.3).
4. **Cauchy decrease** (Theorem 7), with no rank assumption, no threshold on ε, and no
   dimension constant:
   $$-q(s^C)\;\ge\;\frac{\kappa_G}{2}\,h(x)\,\min\!\Big(\Delta,\ \frac{\kappa'\,h(x)}{1+\|H\|}\Big),\qquad \kappa'=\min(1,4(1-\kappa_G)).$$
5. **Convergence.** $\liminf_k h(x_k)=0$ with the proof of `theo:inner_convergence`
   unchanged. Since $h$ is continuous, the growth assumption on $b_k$ also gives
   $\lim_k h(x_k)=0$ (Theorem 9).
6. **Stopping test.** If $h(x)\le\varepsilon$, then
   $\pi_\varepsilon(x)\le\max(1,1/\sigma^+)\,h(x)$. Here $\sigma^+$ is the smallest nonzero
   singular value of the matrices $[A;E_S^\top]$, and the multipliers are those of minimal
   sign violation, which is what the code computes. Hence the test
   $\pi_\varepsilon(x_k)\le\omega$ is met after finitely many iterations, for every
   $\varepsilon>0$ (Corollary 12). The outer analysis does not change.

---

## 1. Setting

### 1.1 Step set, path, model

Fix an inner iterate $x\in\Omega$, its gradient $g$, a symmetric $H$ and $\Delta>0$. The steps
lie in the polyhedron

$$D=\{s\in\mathbb R^n \mid As=0,\ \ell^\Delta\le s\le u^\Delta\},\qquad \ell^\Delta_i=\max(-\Delta,\,l_i-x_i)\le0,\quad u^\Delta_i=\min(\Delta,\,u_i-x_i)\ge0,$$

so $0\in D$, $D$ is bounded, and $D=(\Omega-x)\cap\{\|s\|_\infty\le\Delta\}$. The projected gradient path is $s(t)=P_D(-tg)$, $t\ge0$, and

$$\psi(t)=q(s(t))=\langle g,s(t)\rangle+\tfrac12\langle s(t),Hs(t)\rangle,\qquad L(t)=-\langle g,s(t)\rangle .$$

$L$ is the decrease of the *linear* model along the path, and
$\psi(t)=-L(t)+\tfrac12\langle s(t),Hs(t)\rangle$. The criticality measure is the projected
gradient

$$p=P_\Omega(x-g)-x,\qquad h(x)=\|p\| .$$

It is continuous in $x$, and $h(x)=0$ if and only if $x$ is first-order critical. We also
mention the measure
$\chi(x)=-\min\{\langle g,d\rangle\mid x+d\in\Omega,\ \|d\|_\infty\le1\}$ of Conn, Gould and
Toint (*Trust-Region Methods*), which is not needed in the proofs.

### 1.2 Faces, normal cones and projections

This subsection gathers the facts on polyhedra that the description of the path uses. Since
$\Delta<\infty$, all the bounds of $D$ are finite. We write them as pairs: $\beta=(i,-)$ stands
for $s_i\ge\ell^\Delta_i$ and $\beta=(i,+)$ for $s_i\le u^\Delta_i$. With

$$a_{(i,-)}=-e_i,\quad b_{(i,-)}=-\ell^\Delta_i,\qquad a_{(i,+)}=e_i,\quad b_{(i,+)}=u^\Delta_i,$$

the bound $\beta$ reads $\langle a_\beta,s\rangle\le b_\beta$, and
$D=\{s\mid As=0,\ \langle a_\beta,s\rangle\le b_\beta\text{ for all }\beta\}$.
- For $s\in D$, $\mathcal A(s)=\{\beta\mid\langle a_\beta,s\rangle=b_\beta\}$ is the set of bounds
  active at $s$.
- For a set $\mathcal W$ of bounds,
  $I(\mathcal W)=\{i\mid(i,-)\in\mathcal W\text{ or }(i,+)\in\mathcal W\}$ is the set of the
  variables concerned, and $E_I$ denotes the matrix of the columns $e_i$, $i\in I$.

**Definition.**
- (a) A *face* of $D$ is a nonempty set of the form
  $$F_{\mathcal W}=\{s\in D\mid\langle a_\beta,s\rangle=b_\beta\text{ for all }\beta\in\mathcal W\},$$
  where $\mathcal W$ is a set of bounds. For a face $F$,
  $\mathcal W(F)=\{\beta\mid\langle a_\beta,s\rangle=b_\beta\text{ for all }s\in F\}$ is the set of
  the bounds active on all of $F$, and $I(F)=I(\mathcal W(F))$.
- (b) The *affine hull* of a nonempty set $S\subseteq\mathbb R^n$ is the smallest affine subspace
  of $\mathbb R^n$ containing $S$, that is, the set of its affine combinations
  $$\operatorname{aff}S=\Big\{\sum_{k=1}^p\lambda_ks_k\ \Big|\ p\ge1,\ s_k\in S,\ \lambda_k\in\mathbb R,\ \sum_{k=1}^p\lambda_k=1\Big\}.$$
  An affine subspace $E$ is the translate $E=c+T$ of a unique linear subspace $T=E-E$, its
  *direction space*, by any $c\in E$.
- (c) The *relative interior* of a convex set $S$ is its interior relative to
  $\operatorname{aff}S$:
  $$\operatorname{ri}S=\big\{s\in S\ \big|\ \exists r>0,\ \{v\in\operatorname{aff}S\mid\|v-s\|<r\}\subseteq S\big\}.$$
- (d) The *normal cone* to $D$ at $s\in D$ is
  $N_D(s)=\{w\in\mathbb R^n\mid\langle w,v-s\rangle\le0\text{ for all }v\in D\}$. As an
  intersection of closed half-spaces, it is a closed convex cone.

A face $F$ is convex, as the intersection of $D$ with hyperplanes, and $F=F_{\mathcal W(F)}$.
Indeed, $F=F_{\mathcal W}$ for some $\mathcal W\subseteq\mathcal W(F)$, so that
$F_{\mathcal W(F)}\subseteq F_{\mathcal W}=F$, and $F\subseteq F_{\mathcal W(F)}$ by definition
of $\mathcal W(F)$.

**Lemma A (faces).** Let $F$ be a face of $D$, $\mathcal W=\mathcal W(F)$ and $I=I(F)$.
1. The affine hull of $F$ is
   $$\operatorname{aff}F=\{s\in\mathbb R^n\mid As=0,\ \langle a_\beta,s\rangle=b_\beta\text{ for all }\beta\in\mathcal W\},$$
   i.e. the points with $As=0$, $s_i=\ell^\Delta_i$ for $(i,-)\in\mathcal W$ and
   $s_i=u^\Delta_i$ for $(i,+)\in\mathcal W$. Its direction space is
   $T_F=\{v\in\mathbb R^n\mid Av=0,\ v_i=0\text{ for all }i\in I\}$.
2. $\operatorname{ri}F=\{s\in D\mid\mathcal A(s)=\mathcal W\}=\{s\in\operatorname{aff}F\mid\langle a_\beta,s\rangle<b_\beta\text{ for all }\beta\notin\mathcal W\}$.
   In particular, $\operatorname{ri}F$ is convex and nonempty.
3. Every $s\in D$ belongs to $\operatorname{ri}F$ for exactly one face $F$, namely
   $F=F_{\mathcal A(s)}$, and then $\mathcal W(F)=\mathcal A(s)$.
4. $D$ has finitely many faces.

*Proof.*
- **A point of $F$ with active set $\mathcal W$.** For each bound $\beta\notin\mathcal W$, there
  is $s^\beta\in F$ with $\langle a_\beta,s^\beta\rangle<b_\beta$, by definition of $\mathcal W$.
  Let $\hat s$ be the average of these points (any point of $F$ if every bound is in
  $\mathcal W$). Then $\hat s\in F$ by convexity. For $\beta\notin\mathcal W$,
  $\langle a_\beta,\hat s\rangle$ is the average of numbers $\le b_\beta$, one of which is
  $<b_\beta$, so $\langle a_\beta,\hat s\rangle<b_\beta$. The bounds of $\mathcal W$ are active
  at $\hat s$ since $\hat s\in F$.
- **1.** Let $E$ be the right-hand side. It is an affine subspace containing $F$, so
  $\operatorname{aff}F\subseteq E$. Conversely, let $e\in E$ and $s_\lambda=\hat s+\lambda(e-\hat s)$
  for $\lambda>0$.
  - $s_\lambda\in E$, so $As_\lambda=0$ and the bounds of $\mathcal W$ are active at $s_\lambda$.
  - The finitely many strict inequalities $\langle a_\beta,\hat s\rangle<b_\beta$,
    $\beta\notin\mathcal W$, still hold at $s_\lambda$ for $\lambda$ small enough.
  - Hence $s_\lambda\in D$ and $s_\lambda\in F$. Then
    $e=(1-1/\lambda)\,\hat s+(1/\lambda)\,s_\lambda$ is an affine combination of points of $F$,
    and $e\in\operatorname{aff}F$.

  The direction space is
  $E-E=\{v\mid Av=0,\ \langle a_\beta,v\rangle=0\ (\beta\in\mathcal W)\}$, and
  $\langle a_\beta,v\rangle=\pm v_i$ for $\beta=(i,\pm)$, which gives $T_F$.
- **2.** Let $s\in D$ with $\mathcal A(s)=\mathcal W$. Then $s\in F_{\mathcal W}=F$. For
  $v\in\operatorname{aff}F=E$ close to $s$, the strict inequalities
  $\langle a_\beta,v\rangle<b_\beta$, $\beta\notin\mathcal W$, still hold, and the other
  constraints of $D$ hold with equality on $E$. So $v\in F$, and $s\in\operatorname{ri}F$.
  Conversely, let $s\in F$ and suppose that a bound $\beta\notin\mathcal W$ is active at $s$.
  For $\lambda>0$, the point $s+\lambda(s-\hat s)$ belongs to $\operatorname{aff}F$, tends to
  $s$ as $\lambda\to0$, and violates $\beta$, since
  $\langle a_\beta,s+\lambda(s-\hat s)\rangle=b_\beta+\lambda\big(b_\beta-\langle a_\beta,\hat s\rangle\big)>b_\beta$.
  So $s\notin\operatorname{ri}F$. This proves the first equality. For the second, a point of
  $\operatorname{aff}F$ satisfies $As=0$ and the bounds of $\mathcal W$ with equality. It thus
  belongs to $D$ with $\mathcal A(s)=\mathcal W$ if and only if the other bounds hold strictly.
  This set is the intersection of an affine subspace with open half-spaces, hence convex, and it
  contains $\hat s$.
- **3.** Let $s\in D$ and $F=F_{\mathcal A(s)}$, a face since $s\in F$.
  - Every $s'\in F$ satisfies the bounds of $\mathcal A(s)$ with equality, so
    $\mathcal A(s)\subseteq\mathcal W(F)$.
  - Conversely, the bounds of $\mathcal W(F)$ are active at $s\in F$, so
    $\mathcal W(F)\subseteq\mathcal A(s)$.
  - Hence $\mathcal W(F)=\mathcal A(s)$, and $s\in\operatorname{ri}F$ by 2.
  - If $s\in\operatorname{ri}F'$ for another face $F'$, then $\mathcal W(F')=\mathcal A(s)$ by 2,
    and $F'=F_{\mathcal W(F')}=F_{\mathcal A(s)}=F$.
- **4.** There are finitely many sets of bounds. ∎

**Lemma B (normal cones).** Let $s\in D$.
1. $N_D(s)=\big\{A^\top\xi+\sum_{\beta\in\mathcal A(s)}\lambda_\beta a_\beta\ \big|\ \xi\in\mathbb R^m,\ \lambda_\beta\ge0\big\}$.
2. Equivalently, $w\in N_D(s)$ if and only if
   $-w=A^\top\xi+\sum_{i\in I(\mathcal A(s))}\eta_ie_i$ for some $\xi\in\mathbb R^m$ and $\eta$
   with $\sigma_i\eta_i\ge0$. Here $\sigma_i=+1$ if only the lower bound of $i$ is active,
   $\sigma_i=-1$ if only the upper bound is, and $\sigma_i=0$ (no condition) if both are, i.e.
   $\ell^\Delta_i=u^\Delta_i$.
3. For a face $F$, $N_D(s)$ is the same cone at every $s\in\operatorname{ri}F$. We denote it by
   $N_D(F)$. It is contained in $T_F^\perp=\operatorname{range}[A^\top\ E_{I(F)}]$.

*Proof.*
- **1.**
  - If $w=A^\top\xi+\sum_{\beta\in\mathcal A(s)}\lambda_\beta a_\beta$ with $\lambda\ge0$, then
    for $v\in D$,
    $\langle w,v-s\rangle=\langle\xi,A(v-s)\rangle+\sum_{\beta\in\mathcal A(s)}\lambda_\beta\big(\langle a_\beta,v\rangle-b_\beta\big)\le0$,
    since $Av=As=0$ and $\langle a_\beta,s\rangle=b_\beta$ for $\beta\in\mathcal A(s)$.
  - Conversely, let $w\in N_D(s)$. Then $s$ solves the linear program
    $\max\{\langle w,v\rangle\mid Av=0,\ \langle a_\beta,v\rangle\le b_\beta\text{ for all }\beta\}$.
    By the duality theorem of linear programming, its dual
    $\min\{\sum_\beta b_\beta\lambda_\beta\mid A^\top\xi+\sum_\beta\lambda_\beta a_\beta=w,\ \lambda\ge0\}$
    has a solution $(\xi,\lambda)$ with the same value $\langle w,s\rangle$.
  - Subtracting,
    $0=\sum_\beta b_\beta\lambda_\beta-\langle w,s\rangle=\sum_\beta\lambda_\beta\big(b_\beta-\langle a_\beta,s\rangle\big)$,
    a sum of nonnegative terms. So $\lambda_\beta=0$ for every $\beta\notin\mathcal A(s)$.
- **2.** Group the two bounds of each variable:
  $\sum_{\beta\in\mathcal A(s)}\lambda_\beta a_\beta=\sum_i(\lambda_{(i,+)}-\lambda_{(i,-)})\,e_i$,
  with $\lambda_{(i,\pm)}=0$ when $(i,\pm)$ is inactive. Set
  $\eta_i=\lambda_{(i,-)}-\lambda_{(i,+)}$ and change $\xi$ into $-\xi$. Then $\eta_i\ge0$ if
  only the lower bound is active, $\eta_i\le0$ if only the upper bound is, and $\eta_i$ is
  arbitrary if both are. Conversely, every such $\eta$ is obtained in this way.
- **3.** By 1, $N_D(s)$ only depends on $\mathcal A(s)$, which equals $\mathcal W(F)$ on
  $\operatorname{ri}F$ by Lemma A.2. By Lemma A.1,
  $T_F=\{v\mid Av=0,\ E_{I(F)}^\top v=0\}$ is the null space of the matrix whose transpose is
  $[A^\top\ E_{I(F)}]$, so $T_F^\perp$ is the range of $[A^\top\ E_{I(F)}]$. By 2, every element
  of $N_D(F)$ lies in this range. ∎

**Lemma C (projections).** Let $y\in\mathbb R^n$ and let $F$ be a face of $D$.
1. $s=P_D(y)$ if and only if $s\in D$ and $y-s\in N_D(s)$. Moreover,
   $\|P_D(y)-P_D(y')\|\le\|y-y'\|$ for all $y'\in\mathbb R^n$.
2. $P_D(y)\in\operatorname{ri}F$ if and only if
   $y\in C_F:=\operatorname{ri}F+N_D(F)=\{s+w\mid s\in\operatorname{ri}F,\ w\in N_D(F)\}$. In any
   decomposition $y=s+w$ with $s\in\operatorname{ri}F$ and $w\in N_D(F)$, necessarily
   $s=P_D(y)$.
3. If $y\in C_F$, then $P_D(y)=P_{\operatorname{aff}F}(y)=c_F+P_{T_F}\,y$, where
   $c_F=P_{\operatorname{aff}F}(0)\in T_F^\perp$.
4. $C_F$ is convex.

*Proof.*
- **1.** Let $f(v)=\frac12\|v-y\|^2$, which is strictly convex, so that $P_D(y)$ is its unique
  minimizer on $D$. For $s,v\in D$, $f(v)=f(s)+\langle s-y,v-s\rangle+\frac12\|v-s\|^2$.
  - If $\langle y-s,v-s\rangle\le0$ for all $v\in D$, i.e. $y-s\in N_D(s)$, then $f(v)\ge f(s)$
    on $D$.
  - Conversely, if $s$ minimizes $f$ on $D$, then
    $0\le f(s+\tau(v-s))-f(s)=\tau\langle s-y,v-s\rangle+\frac{\tau^2}2\|v-s\|^2$ for $v\in D$
    and $\tau\in(0,1]$, by convexity of $D$. Dividing by $\tau$ and letting $\tau\to0$ gives
    $\langle y-s,v-s\rangle\le0$.
  - For the Lipschitz bound, write $s=P_D(y)$ and $s'=P_D(y')$. Adding the inequalities
    $\langle y-s,s'-s\rangle\le0$ and $\langle y'-s',s-s'\rangle\le0$ gives
    $\|s-s'\|^2\le\langle y-y',s-s'\rangle\le\|y-y'\|\,\|s-s'\|$.
- **2.** If $s=P_D(y)\in\operatorname{ri}F$, then $w=y-s\in N_D(s)=N_D(F)$ by 1 and Lemma B.3,
  and $y=s+w\in C_F$. Conversely, let $y=s+w$ with $s\in\operatorname{ri}F$ and $w\in N_D(F)$.
  Then $s\in D$ and $y-s=w\in N_D(s)$, so $s=P_D(y)$ by 1, and $P_D(y)\in\operatorname{ri}F$.
  This also proves the uniqueness of the decomposition.
- **3.** Let $E=\operatorname{aff}F=c+T_F$.
  - For $s\in E$ with $y-s\in T_F^\perp$ and any $e\in E$, we have $s-e\in T_F$, hence
    $\|y-e\|^2=\|y-s\|^2+\|s-e\|^2$. So $P_E(y)$ is the unique $s\in E$ with
    $y-s\in T_F^\perp$.
  - If $y\in C_F$, then $s=P_D(y)$ belongs to $\operatorname{ri}F\subseteq E$ and
    $y-s\in N_D(F)\subseteq T_F^\perp$ by 2 and Lemma B.3, so $P_D(y)=P_E(y)$.
  - For $y=0$, the characterization gives $c_F=P_E(0)\in T_F^\perp$. Then, for any $y$,
    $c_F+P_{T_F}y\in E$ and $y-c_F-P_{T_F}y=(y-P_{T_F}y)-c_F\in T_F^\perp$, so
    $P_E(y)=c_F+P_{T_F}y$.
- **4.** $\operatorname{ri}F$ is convex by Lemma A.2, and $N_D(F)$ is a convex cone. The sum of
  two convex sets is convex:
  $\lambda(s_1+w_1)+(1-\lambda)(s_2+w_2)=\big(\lambda s_1+(1-\lambda)s_2\big)+\big(\lambda w_1+(1-\lambda)w_2\big)$
  for $\lambda\in[0,1]$. ∎

With $y=-tg$, Lemmas C.1 and B.2 characterize the path: $s(t)\in D$ and

$$tg+s(t)=A^\top\xi(t)+\textstyle\sum_i\eta_i(t)\,e_i,\qquad \eta_i(t)\ge0 \text{ if } s_i(t)=\ell^\Delta_i<u^\Delta_i,\quad \eta_i(t)\le0 \text{ if } s_i(t)=u^\Delta_i>\ell^\Delta_i,\quad \eta_i(t)=0 \text{ if } \ell^\Delta_i<s_i(t)<u^\Delta_i, \tag{1.1}$$

with any sign if $\ell^\Delta_i=u^\Delta_i$. These are the KKT conditions of
$\min_{s\in D}\tfrac12\|s+tg\|^2$. With $\sigma_i=+1$ at a lower and $\sigma_i=-1$ at an upper
bound, the sign condition reads $\sigma_i\eta_i\ge0$, as in §2.3 of the plan.

### 1.3 Pieces, events and multipliers

**Proposition 1 (structure of the path).**
1. $[0,\infty)$ is partitioned into the sets $I_F=\{t\ge0\mid s(t)\in\operatorname{ri}F\}$,
   $F$ ranging over the faces of $D$, and each $I_F$ is an interval.
2. On $I_F$, $s(t)=P_{\operatorname{aff}F}(-tg)=c_F-t\,z_F$ with $z_F=P_{T_F}\,g$, so
   $s'(t)=-z_F$ in the interior of $I_F$. The path is continuous and piecewise affine with
   finitely many pieces $0=t_0<t_1<\dots<t_N=T_{\rm end}$, and it is constant on
   $[T_{\rm end},\infty)$.
3. A piece on $\operatorname{ri}F$ ends at a *hit*, when a bound $\beta\notin\mathcal W(F)$
   becomes active (a free variable reaches a bound), or at a *release*, when no multiplier vector
   of the right signs represents $tg+s(t)$ any more. When the rows of $A$ and the $e_i^\top$,
   $i\in I(F)$, are linearly independent, a release means that a multiplier
   $\eta_j(t)=\eta_j(t_k)+(t-t_k)\theta_j$, affine with slope $\theta_j$ (the bound part of the
   decomposition $g=z_F+A^\top y+E_{I(F)}\theta$), crosses zero with $\sigma_j\theta_j<0$.

*Proof.*
- **1. Partition.** For $t\ge0$, $s(t)\in D$ belongs to $\operatorname{ri}F$ for exactly one
  face $F$ by Lemma A.3, that is, $t$ belongs to exactly one set $I_F$.
- **1. Intervals.** By Lemma C.2 with $y=-tg$, $I_F=\{t\ge0\mid-tg\in C_F\}$. Let
  $t_1,t_2\in I_F$ and $t=\lambda t_1+(1-\lambda)t_2$ with $\lambda\in[0,1]$. Since
  $t\mapsto-tg$ is linear, $-tg=\lambda(-t_1g)+(1-\lambda)(-t_2g)$, which belongs to $C_F$ by
  convexity (Lemma C.4). Hence $t\in I_F$. A convex subset of $\mathbb R$ is an interval, which
  may be empty or reduced to a point.
- **2. Affine on each $I_F$.** Lemma C.3 with $y=-tg$ gives $s(t)=c_F-tP_{T_F}g$ for
  $t\in I_F$.
- **2. Finitely many pieces.** By Lemma A.4, the nonempty sets $I_F$ are finitely many pairwise
  disjoint intervals covering $[0,\infty)$. Let $0=t_0<t_1<\dots<t_N$ be their finite endpoints.
  An open interval $(t_k,t_{k+1})$, or $(t_N,\infty)$, contains no endpoint, so it meets a single
  $I_F$: two disjoint intervals meeting it would have an endpoint inside it. Hence $s$ is affine
  on each of these open intervals.
- **2. Continuity.** By Lemma C.1, $\|s(t)-s(t')\|\le|t-t'|\,\|g\|$, so $s$ is also affine on
  their closures.
- **2. End of the path.** On $[t_N,\infty)$, $s$ is affine and bounded, since $D$ is bounded, so
  its slope vanishes. We set $T_{\rm end}=t_N$.
- **3. A criterion for $I_F$.** Let $F$ be a face such that $I_F$ has a nonempty interior and a
  finite right endpoint $\bar t$. For all $t\in\mathbb R$, set $\hat s(t)=c_F-tz_F$ and
  $\hat w(t)=-tg-\hat s(t)$. We claim that, for $t\ge0$,
  $$t\in I_F\iff \hat s(t)\in\operatorname{ri}F\ \text{ and }\ \hat w(t)\in N_D(F).$$
  If the right-hand side holds, then $-tg=\hat s(t)+\hat w(t)\in C_F$, and $t\in I_F$ by
  Lemma C.2. Conversely, if $t\in I_F$, then $s(t)=\hat s(t)\in\operatorname{ri}F$ by 2, and
  $\hat w(t)=-tg-s(t)\in N_D(F)$ by the uniqueness in Lemma C.2.
- **3. The two conditions.** Since $c_F\in\operatorname{aff}F$ and $z_F\in T_F$,
  $\hat s(t)\in\operatorname{aff}F$ for all $t$. So, by Lemma A.2, the first condition reads
  $\langle a_\beta,\hat s(t)\rangle<b_\beta$ for all $\beta\notin\mathcal W(F)$: finitely many
  strict affine inequalities in $t$. The second condition defines a closed set of values of $t$,
  since $N_D(F)$ is closed and $\hat w$ is continuous. For $t>\bar t$ arbitrarily close to
  $\bar t$, $t\notin I_F$, so one of the two conditions fails.
  - *Hit.* If the first condition fails, there is a bound $\beta\notin\mathcal W(F)$, one of
    finitely many, with $\langle a_\beta,\hat s(t)\rangle\ge b_\beta$ for $t>\bar t$
    arbitrarily close to $\bar t$. Since $\langle a_\beta,\hat s(t)\rangle<b_\beta$ in the
    interior of $I_F$, continuity gives $\langle a_\beta,s(\bar t)\rangle=b_\beta$: the bound
    $\beta$, inactive along the piece, becomes active at $\bar t$.
  - *Release.* If the second condition fails, $\hat w(t)\notin N_D(F)$ for $t>\bar t$
    arbitrarily close to $\bar t$, while $\hat w(\bar t)\in N_D(F)$ since $N_D(F)$ is closed.
    Now $-\hat w(t)=tg+\hat s(t)=t(g-z_F)+c_F$, and both $g-z_F=(I-P_{T_F})g$ and $c_F$ belong
    to $T_F^\perp=\operatorname{range}[A^\top\ E_{I(F)}]$. So $-\hat w(t)$ can always be
    written $A^\top\xi+E_{I(F)}\eta$, but, by Lemma B.2, for $t>\bar t$ close to $\bar t$, none
    of these representations has $\sigma_i\eta_i\ge0$ for all $i\in I(F)$.

  Both events may occur at the same $\bar t$, which is then a degenerate breakpoint.
- **3. Independent rows.** Suppose that the rows of $A$ and the $e_i^\top$, $i\in I(F)$, are
  linearly independent. Then the coefficients $(\xi(t),\eta(t))$ of $-\hat w(t)$ are unique, and
  they are affine in $t$ since $-\hat w$ is affine. With the unique decomposition
  $g-z_F=A^\top y+E_{I(F)}\theta$, they read $\eta(t)=\eta(\bar t)+(t-\bar t)\,\theta$.
  - If $\sigma_i\eta_i(\bar t)>0$, this sign persists for $t$ close to $\bar t$.
  - If $\sigma_i\eta_i(\bar t)=0$ and $\sigma_i\theta_i\ge0$, then $\sigma_i\eta_i(t)\ge0$ for
    $t>\bar t$.
  - Hence $\hat w(t)$ leaves $N_D(F)$ right after $\bar t$ if and only if
    $\sigma_j\eta_j(\bar t)=0$ and $\sigma_j\theta_j<0$ for some $j\in I(F)$: the multiplier of
    the bound of $j$ crosses zero, which is the release of that bound. ∎

At a breakpoint, the right derivative is $s'(t^+)=P_{K}(-g)$ with
$K=T_D(s)\cap\{tg+s\}^\perp$ (Haraux). This is what `local_cone_projection!` computes. The
proofs below only use Proposition 1.

### 1.4 Speed along the path

On a piece, $\|s'(t)\|=\|z_F\|$ is the norm of the reduced gradient.

**Lemma 2 (hits slow down, releases speed up).** Let $F$, $F'$ be the faces of two
consecutive pieces, and write $z=z_F$, $z'=z_{F'}$.
- If $F'\subseteq F$ (a hit), then $T_{F'}\subseteq T_F$ and $\|z'\|\le\|z\|$.
- If $I(F')=I(F)\setminus\{j\}$ (a single release, independent rows), then
  $$\|z'\|^2=\|z\|^2+\theta_j^2\,\|P_{T_{F'}}e_j\|^2\ \ge\ \|z\|^2 .$$

*Proof.*
- If $F'\subseteq F$, then $\mathcal W(F')\supseteq\mathcal W(F)$, so $T_{F'}\subseteq T_F$ by
  Lemma A.1, and the first claim holds because $z'=P_{T_{F'}}z$.
- For the second, decompose $g=z+A^\top y+E_{I(F)}\theta$. The vectors $A^\top y$ and $e_k$,
  $k\ne j$, are orthogonal to $T_{F'}$, and $P_{T_{F'}}z=z$, so $z'=z+\theta_jP_{T_{F'}}e_j$.
- Moreover $P_{T_{F'}}e_j\perp T_F$, since its component in $T_F$ is $P_{T_F}e_j=0$, so the two
  terms are orthogonal. ∎

On a box, a coordinate that reaches a bound stays there, so every breakpoint is a hit and
$\|s'\|$ is nonincreasing. Over a polyhedron, the dual events make $\|s'\|$ jump up. A
stationary piece followed by a release (as in the paper's example: $0\to\sqrt2/10$) is the
extreme case.

---

## 2. Path inequalities

**Lemma 3.** For all $t>0$:
- (a) $L(t)=\int_0^t\|s'(\tau)\|^2\,d\tau$. Hence $L$ is continuous, nondecreasing, and
  constant on $[T_{\rm end},\infty)$, where it equals $\max_{t}L$.
- (b) $\|s(t)\|^2\le t\,L(t)$.
- (c) $L(t)\ge-\langle g,v\rangle-\dfrac{\|v\|^2}{4t}$ for every $v\in D$.
- (d) $L(t)\ge\dfrac{h^2}{2}\min\!\Big(1,\dfrac{\Delta}{h},2t\Big)=\dfrac h2\min(h,\Delta,2th)$, with $h=h(x)$.

*Proof.*
- (a) On a piece, $-\langle g,s'\rangle=\langle g,P_{T_F}g\rangle=\|P_{T_F}g\|^2=\|s'\|^2$,
  and $L(0)=0$.
- (b) Take the inner product of (1.1) with $s(t)$. Since $As=0$,
  $\langle tg+s,s\rangle=\sum_i\eta_is_i$. Each term is $\le0$: at a lower bound
  $s_i=\ell^\Delta_i\le0$ and $\eta_i\ge0$, at an upper bound $s_i=u^\Delta_i\ge0$ and
  $\eta_i\le0$, and $s_i=0$ for a fixed variable. Hence $t\langle g,s\rangle\le-\|s\|^2$.
  This is also the variational inequality of the projection with $v=0$.
- (c) The projection gives $\langle-tg-s,v-s\rangle\le0$, i.e.
  $t\langle g,s\rangle\le t\langle g,v\rangle+\langle s,v\rangle-\|s\|^2\le t\langle g,v\rangle+\|v\|^2/4$.
- (d) Let $c=\min(1,\Delta/\|p\|_\infty)\ge\min(1,\Delta/h)$ and $v=\lambda cp$ with
  $\lambda\in[0,1]$. Then $v\in D$, because $\Omega-x$ is convex and contains 0, and
  $\|v\|_\infty\le\Delta$. The projection $p$ satisfies $\langle g,p\rangle\le-\|p\|^2=-h^2$
  (variational inequality with $y=x$). Then (c) gives
  $L(t)\ge\lambda ch^2-\lambda^2c^2h^2/(4t)$. Take $\lambda=1$ if $2t\ge c$ and
  $\lambda=2t/c$ otherwise. ∎

Item (d) deals with open point 2 of the plan: merging the $\ell_\infty$ trust region into the
bounds costs nothing, since a scaled projected gradient step stays in $D$.

Lemma 3(a) shows why the decrease is governed by the *integral* of $\|s'\|^2$. The pointwise
value of the reduced gradient at early times does not control it, unless $\|s'\|$ is monotone.

---

## 3. The first local minimizer

Scan the pieces in order. On a piece $[t_i,t_{i+1}]$ with $d=s'\ne0$, let
$\varphi_1=\langle g+Hs(t_i),d\rangle=\psi'(t_i^+)$ and $\varphi_2=\langle d,Hd\rangle$. Stop
at $t_i$ if $\varphi_1\ge0$, and at $t_i-\varphi_1/\varphi_2$ if $\varphi_1<0<\varphi_2$ and
this point lies before $t_{i+1}$. Otherwise, go to the next piece. Stationary pieces ($d=0$)
are skipped, and the scan stops at $T_{\rm end}$ at the latest. Call $t_1$ the stopping point.
By construction:

> $\psi$ is nonincreasing on $[0,t_1]$. If $t_1<T_{\rm end}$, the direction $d\ne0$ at $t_1$
> (of the current piece, or of the next nonstationary one) satisfies $\langle g+Hs(t_1),d\rangle\ge0$.
> Hence $\|d\|^2=-\langle g,d\rangle\le\langle Hs(t_1),d\rangle$, that is
> $$\|d\|\le\|H\|\,\|s(t_1)\|. \tag{3.1}$$

So an early stop requires a slow direction.

### 3.1 Without releases: the box argument is correct

**Proposition 4.** Suppose $\|s'\|$ is nonincreasing on $(t_1,\infty)$, i.e. no bound is
released after $t_1$. This holds in particular when $\Omega$ is a box, with the trust region
included in the bounds. Then
$$-\psi(t_1)\ \ge\ \tfrac3{10}\,h\,\min\!\Big(\Delta,\ \frac h{1+\|H\|}\Big).$$

*Proof.* Let $\tau_0=1/(2(1+\|H\|))$, so that $\|H\|\tau_0\le\frac12$. By Lemma 3(b), for
$t\le\tau_0$, $-\psi(t)\ge L(t)(1-\tfrac12\|H\|t)\ge\tfrac34L(t)$.
- If $t_1\ge\tau_0$: $\psi$ is nonincreasing on $[0,t_1]$, so
  $-\psi(t_1)\ge-\psi(\tau_0)\ge\frac34L(\tau_0)$.
- If $t_1=T_{\rm end}<\tau_0$: $-\psi(t_1)\ge\frac34L(T_{\rm end})\ge\frac34L(\tau_0)$.
- Otherwise $t_1<\min(\tau_0,T_{\rm end})$. By monotonicity of $\|s'\|$ and (3.1),
  $$L(\tau_0)-L(t_1)=\int_{t_1}^{\tau_0}\|s'\|^2\le\tau_0\|d\|^2\le\tau_0\|H\|^2\|s(t_1)\|^2\le\tau_0\|H\|^2t_1L(t_1)\le\tfrac14L(t_1),$$
  so $-\psi(t_1)\ge\frac34L(t_1)\ge\frac35L(\tau_0)$.

In all cases $-\psi(t_1)\ge\frac35L(\tau_0)$. Conclude with Lemma 3(d) and $2\tau_0h=h/(1+\|H\|)$. ∎

The monotonicity of the reduced gradient, the core of Conn–Gould–Toint 1988 and of the old
draft, is used in exactly one place: to stop the path from speeding up after $t_1$. This is
where releases break the argument.

### 3.2 With releases: a counterexample

Take the paper's polyhedron plus a free variable:

$$\Omega=\{x\in\mathbb R^4\mid x_1+x_2+x_3=0,\ x_1\ge-1,\ x_2\le1\},\qquad g=(0.3,\,-0.8,\,0.5,\,a),\qquad H=vv^\top,\ v=e_4-e_2,$$

with $\Delta=1$, $x_\delta=(-1,1,0,0)+\delta(1,-3,2,0)$ and $0<a\le\delta\le0.01$. Then $H\succeq0$
and $\|H\|=2$. Since $D=D_3\times[-1,1]$, the path is the product of the 3D path of the paper's
zigzag example and $s_4(t)=-at$:

| $t$ | direction $s'$ | event at the end |
|---|---|---|
| $[0,\ 10\delta/3]$ | $-g$ | hit $x_1\ge-1$ |
| $[10\delta/3,\ 50\delta/13]$ | $(0,0.65,-0.65,-a)$ | hit $x_2\le1$: vertex |
| $[50\delta/13,\ 5\delta]$ | $(0,0,0,-a)$, **slow** | release of $x_1\ge-1$ |
| $[5\delta,\ \approx10]$ | $(0.1,0,-0.1,-a)$ | trust region |

At the vertex, $g=(0,0,0,a)+0.5\,(1,1,1,0)-0.2\,e_1-1.3\,e_2$. The multiplier of $x_1\ge-1$
has slope $\theta_1=-0.2<0$: it decreases from $0.45\cdot20\delta/39$ and vanishes at
$t=5\delta$. The speed jumps from $a$ to $\sqrt{0.02+a^2}$ (Lemma 2).

- **Before the vertex** the scan does not stop. On the first two pieces
  $\varphi_1\le-0.845+1.8\delta<0$, and the minimizers of the quadratics lie beyond the
  pieces.
- **At the vertex** $t_v=50\delta/13$, $s=(-\delta,3\delta,-2\delta,-at_v)$, and on the slow
  piece
  $$\psi'(t_v^+)=-a^2+\langle v,s\rangle\langle v,d\rangle=a\,(3\delta+at_v-a)>0 .$$
  So $t_1=t_v$, and the decrease is
  $$-\psi(t_1)=L(t_v)-\tfrac12\langle v,s\rangle^2=3.7\,\delta+\tfrac{50}{13}a^2\delta-\tfrac12\big(3\delta+\tfrac{50}{13}a\delta\big)^2=3.7\,\delta+O(\delta^2).$$
- **The measures stay bounded away from zero.**
  - $h(x_\delta)\to\sqrt{0.02+a^2}\ge\sqrt2/10$.
  - $\chi(x_\delta)\to0.2$.
  - $\pi_\varepsilon(x_\delta)=\max(a,0.2)=0.2$ as soon as both bounds are ε-active.
- **After the release** $\psi$ keeps decreasing by about $0.2$ up to the trust region
  ($\psi'=-(0.02+a^2)+a(3\delta+at)<0$).

**Proposition 5.** No inequality $-q(s^C)\ge\Phi(h(x),\Delta,\|H\|)$, with $\Phi>0$ for
positive arguments, can hold for the first local minimizer over polyhedra. The same holds with
$\pi_\varepsilon$ or $\chi$ in place of $h$, and even for convex models.

Computed with `cauchy_step!` (current code), $a=\delta/2$. The last column is the best value
of $\psi$ on $[0,\hat t]$ for $\kappa_G=\frac12$, evaluated on the traced path:

| $\delta$ | $-q(s^C)$, code | $t_1$ | $h(x_\delta)$ | $\chi(x_\delta)$ | $-q$, safeguarded (§4) |
|---|---|---|---|---|---|
| 1e-1 | 3.20e-1 | 3.85e-1 | 0.397 | 0.580 | 0.326 |
| 1e-2 | 3.66e-2 | 3.85e-2 | 0.146 | 0.238 | 0.230 |
| 1e-3 | 3.70e-3 | 3.85e-3 | 0.1415 | 0.204 | 0.203 |
| 1e-4 | 3.70e-4 | 3.85e-4 | 0.1414 | 0.200 | 0.200 |
| 1e-6 | 3.70e-6 | 3.85e-6 | 0.1414 | 0.200 | 0.200 |

With $a=\delta^2$, the slow piece falls below `tol_dir` once $\delta\le10^{-4}$. The code then
treats it as stationary and skips it, which shows that the slow piece is a perturbation of the
stationary piece of the zigzag example. Skipping stationary pieces handles $a=0$, and the
safeguard below handles $a>0$.

This is a failure of the per-iteration decrease, which the trust-region analysis needs. It is
not a proven failure of the algorithm: here the Cauchy point lands on the vertex, and from
there the next path starts with the release direction. A zigzag sequence would have to repeat
this near ever closer faces. Whether one exists is open, and the safeguard makes the question
moot.

---

## 4. A safeguarded Cauchy point

### 4.1 Definition

Fix $\kappa_G\in(0,1)$. Let

$$t_G=\inf\{t>0\mid\psi(t)\ge-\kappa_G\,L(t)\}\in(0,\infty],\qquad \hat t=\min(t_G,T_{\rm end}).$$

$t_G$ is where the model decrease falls to the fraction $\kappa_G$ of the linear decrease: the
curvature starts to matter. Before $t_G$, $\psi\approx-L$, and a stop there can only come from
a slow piece, by (3.1).

**Condition (C).** $\psi(t^C)\le\psi(\hat t)$.

**Rule.**
- If $t_1\ge\hat t$, take $t^C=t_1$. This is the common case, and the current behavior.
- Otherwise ($t_1$ is *premature*), go on scanning from $\hat t$ to the next stop $t^\sharp$
  (the first $t\ge\hat t$ with $\psi'(t^+)\ge0$ on the next nonstationary piece, or
  $T_{\rm end}$). Take for $t^C$ the best of the stops met on $[0,t^\sharp]$.

The rule satisfies (C):
- In the first case, $\psi$ is nonincreasing on $[0,t_1]\ni\hat t$.
- In the second, $\psi$ decreases on $[\hat t,t^\sharp]$, so
  $\psi(t^C)\le\psi(t^\sharp)\le\psi(\hat t)$.

Any other way of reaching (C) is also fine for the theory.

The test $t_1\ge\hat t$ means that $t_1=T_{\rm end}$ or that
$r(t)=\psi(t)+\kappa_GL(t)\ge0$ for some $t\in(0,t_1]$. On a piece,
$r(t_i+\tau)=r(t_i)+(\varphi_1-\kappa_G\langle g,d\rangle)\,\tau+\tfrac12\varphi_2\tau^2$
(note $L'=-\langle g,d\rangle$), so the scan only needs the running value of $L$, one inner
product it already computes, and the maximum of a quadratic on an interval. This is in the
spirit of the generalized Cauchy point of Conn, Gould and Toint (*Trust-Region Methods*,
§12.2), with a single Goldstein condition: the trust region is part of $D$, so their
condition on $\|s\|$ versus $\Delta$ is replaced by the end of the path.

If $\kappa_G\ge\frac12$, a minimizer inside the first piece is always accepted: there
$\psi(t)=-t\|d\|^2+\frac12t^2\langle d,Hd\rangle$ and $L(t)=t\|d\|^2$, so
$\psi(t_1)=-\frac12L(t_1)\ge-\kappa_GL(t_1)$.

### 4.2 Decrease

**Lemma 6.** If $x$ is not critical, then $t_G\ge2(1-\kappa_G)/\|H\|$ (with $t_G=\infty$ if
$H=0$) and $\psi(\hat t)\le-\kappa_GL(\hat t)$.

*Proof.* If $x$ is not critical, $s(t)\ne0$, hence $L(t)\ge\|s(t)\|^2/t>0$, for all $t>0$. By
Lemma 3(b),
$$\psi(t)+\kappa_GL(t)\le-(1-\kappa_G)L(t)+\tfrac12\|H\|tL(t)<0\quad\text{for } t<2(1-\kappa_G)/\|H\|.$$
For $t<t_G$, $\psi(t)<-\kappa_GL(t)$. Then
- if $\hat t=T_{\rm end}<t_G$, the claim is this inequality at $T_{\rm end}$;
- if $\hat t=t_G<\infty$, continuity gives $\psi(t_G)=-\kappa_GL(t_G)$. ∎

**Theorem 7 (Cauchy decrease).** If $s^C=s(t^C)$ satisfies (C), then
$$-q(s^C)\ \ge\ \frac{\kappa_G}{2}\,h(x)\,\min\!\Big(\Delta,\ \frac{\kappa'\,h(x)}{1+\|H\|}\Big),\qquad\kappa'=\min(1,4(1-\kappa_G)).$$

*Proof.* Assume $h=h(x)>0$ and let $\tilde t=2(1-\kappa_G)/(1+\|H\|)\le t_G$. Since $L$ is
nondecreasing and maximal from $T_{\rm end}$ on, $L(\hat t)\ge L(\tilde t)$ in both cases
$\hat t=t_G$ and $\hat t=T_{\rm end}$. By (C), Lemma 6 and Lemma 3(d),
$$-\psi(t^C)\ge-\psi(\hat t)\ge\kappa_GL(\hat t)\ge\kappa_GL(\tilde t)\ge\frac{\kappa_G h}{2}\min(h,\Delta,2\tilde th),$$
and $\min(h,2\tilde th)\ge\kappa'h/(1+\|H\|)$. ∎

The theorem uses neither the rank of the active rows nor ε, and has no dimension constant.
Only convexity of $D$, $0\in D$, and the piecewise structure are used.

---

## 5. Convergence of the inner iterations

The steps satisfy $s_k\in D_k$ and $q_k(s_k)\le q_k(s_k^C)$ (minor iterations), and $H_k$
changes only after successful iterations. The assumptions are those of the paper:
`assumption:functions_C2`, `assumption:inner_iterates_compact` and
`assumption:hessian_norm_compared_convergence_speed`, with $b_k=1+\max_{i\le k}\|H_i\|$.

**Theorem 8.** If Algorithm 2, with the Cauchy step of §4, generates infinitely many iterates,
then $\liminf_k h(x_k)=0$.

*Proof.* If $h(x_k)\ge\underline h>0$ for $k\ge k_0$, Theorem 7 and $b_k\ge\max(1,1+\|H_k\|)$
give
$$-q_k(s_k)\ge c\min(\Delta_k,\zeta/b_k),\qquad c=\kappa_G\underline h/2,\quad\zeta=\kappa'\underline h .$$
This is inequality `eq:decrease_bk`. The proof of `theo:inner_convergence` uses nothing else
about the Cauchy step, and applies word for word. ∎

**Theorem 9.** Under the same assumptions, $\lim_kh(x_k)=0$.

*Proof* (Conn, Gould & Toint 1988, Theorem 11).
1. **Setup.** Suppose $h(x_{m_i})\ge\epsilon_1>0$ along a subsequence, and let $l_i>m_i$ be
   the first index with $h(x_{l_i})<\epsilon_2=\epsilon_1/2$ (Theorem 8). For the successful
   $k\in[m_i,l_i)$, Theorem 7 gives
   $\varphi_k-\varphi_{k+1}\ge\eta_1\frac{\kappa_G\epsilon_2}2\min(\Delta_k,\kappa'\epsilon_2/b_k)$.
2. **Steps are short.** Since $b_k(\varphi_k-\varphi_{k+1})\to0$, we get $b_k\Delta_k\to0$ on
   these iterations, so the minimum is $\Delta_k$ for $i$ large. Hence
   $\|x_{k+1}-x_k\|\le\sqrt n\,\Delta_k\le C(\varphi_k-\varphi_{k+1})$.
3. **Contradiction.** Summing, $\|x_{m_i}-x_{l_i}\|\le C(\varphi_{m_i}-\varphi_{l_i})\to0$,
   since $(\varphi_k)$ is nonincreasing and bounded below on $\mathcal X$. The map
   $x\mapsto\|P_\Omega(x-\nabla\varphi(x))-x\|$ is uniformly continuous on the compact
   $\mathcal X$, so $h(x_{m_i})-h(x_{l_i})\to0$, which contradicts
   $h(x_{m_i})-h(x_{l_i})\ge\epsilon_1/2$. ∎

The paper now only claims $\liminf\pi_\varepsilon=0$, because $\pi_\varepsilon$ is not
continuous. Theorem 9 gives a full limit for $h$.

---

## 6. Link with the stopping test

Recall $\mathcal N_\varepsilon(x)$, generated by $\pm$ the rows of $A$, by $-e_i$
($i\in\mathcal A^-_\varepsilon(x)$) and by $e_i$ ($i\in\mathcal A^+_\varepsilon(x)$).
$\pi_\varepsilon(x)=\max(\|u\|_\infty,\|\theta^{\rm w}\|_\infty)$, with
$u=P_{T_\varepsilon(x)}g$ and $\theta$ a bound part of a decomposition
$g=u+A^\top\xi+E_S\theta$, $S=\mathcal A_\varepsilon(x)$. When the rows of
$\tilde A_S=[A;E_S^\top]$ are dependent, $\theta$ is chosen to minimize $\|\theta^{\rm w}\|_2$
over all decompositions, which is what `wrong_sign_redundant` does (its parametrization by the
redundant bounds spans the whole null space of $[A^\top\ E_S]$).

**Lemma 10 (link).** If $\|p\|_\infty\le\varepsilon$, then
$\operatorname{dist}(-g,\mathcal N_\varepsilon(x))\le\|p\|=h(x)$.

*Proof.* Let $y=x+p$. If $y_i=l_i$, then $x_i-l_i=-p_i\le\varepsilon$, so
$i\in\mathcal A^-_\varepsilon(x)$, and likewise for the upper bounds. Hence
$\mathcal N_\Omega(y)\subseteq\mathcal N_\varepsilon(x)$. Moreover
$-g-p=(x-g)-y\in\mathcal N_\Omega(y)$. ∎

**Lemma 11.** Let $\sigma^+>0$ be the smallest nonzero singular value of the finitely many
matrices $\tilde A_S$, $S\subseteq\{1,\dots,n\}$. With the multipliers of minimal violation,
for all $x\in\Omega$, $\varepsilon\ge0$, $g$:
$$\pi_\varepsilon(x)\le\max(1,1/\sigma^+)\,\operatorname{dist}(-g,\mathcal N_\varepsilon(x)).$$

*Proof.*
1. **Moreau decomposition.** Write $-g=d+w$ with $d=P_{\mathcal T_\varepsilon}(-g)$,
   $w\in\mathcal N_\varepsilon$ and $\|d\|=\operatorname{dist}$. Then
   $w=A^\top\xi'-E_S\theta^+$, where $\theta^+$ has the right signs (the sign is free for
   fixed variables).
2. **Reduced gradient.** $u=P_{T_\varepsilon}g=-P_{T_\varepsilon}d$, so $\|u\|\le\|d\|$.
3. **Multipliers.** $d-P_{T_\varepsilon}d=\tilde A_S^\top y$ with $y$ of minimal norm, hence
   $y\in\operatorname{range}\tilde A_S$ and $\|y\|\le\|d\|/\sigma^+$. Then
   $g=u+A^\top(-\xi'-y_A)+E_S(\theta^+-y_S)$ is a valid decomposition, and its violation is at
   most $|y_{S,i}|$ componentwise. So the minimal violation satisfies
   $\|\theta^{\rm w}\|_\infty\le\|\theta^{\rm w}\|_2\le\|y_S\|\le\|d\|/\sigma^+$. ∎

The right inequality of `lemma:certification`,
$\operatorname{dist}\le\sqrt n\,\pi_\varepsilon$, holds for any multipliers, so it is
unchanged. The left one (with $\sigma_{\min}$ and Assumption `full_rank_active`) is replaced
by Lemma 11.

**Corollary 12.** For every $\varepsilon>0$ and $\omega>0$, the stopping test
$\pi_\varepsilon(x_k)\le\omega$ of Algorithm 2 is met after finitely many iterations.

*Proof.* By Theorem 8 there is $k$ with
$h(x_k)\le\min(\varepsilon,\omega/\max(1,1/\sigma^+))$. Then
$\|p_k\|_\infty\le h(x_k)\le\varepsilon$, and Lemmas 10–11 give
$\pi_\varepsilon(x_k)\le\max(1,1/\sigma^+)\,h(x_k)\le\omega$. ∎

The only requirement on ε is $\varepsilon>0$. The constant does not depend on ε, so
$\varepsilon_K=\min(\varepsilon_{\max},\omega_K)$ works for any $\varepsilon_{\max}>0$. The
outer analysis (§3.2 of the paper) is unchanged.

---

## 7. Consequences

### 7.1 Paper (`sn-preprint-theorie.tex`)

- **§2.1.** Remove Assumption `full_rank_active` and the threshold $\bar\varepsilon$, and
  define the multipliers of minimal violation.
- **§2.3.** The Cauchy step is the first local minimizer along the exact path
  `eq:projected_gradient_path`, with the safeguard of §4.1 (one more constant, $\kappa_G$).
- **§3.1.** Replace `lemma:eps_threshold`, `prop:computed_path`,
  `lemma:initial_direction` and `theo:cauchy_decrease` by Proposition 1, Lemmas 2–3,
  Lemma 6 and Theorem 7.
  - Proposition 4 (box) and the example of §3.2 can replace the zigzag remark. They explain
    why stationary pieces are skipped, and why the first local minimizer needs the
    safeguard over a polyhedron and not over a box.
  - `theo:inner_convergence` keeps its proof, with $h$ in place of $\kappa_0\pi_\varepsilon$.
    Add Theorem 9, Lemma 10 and Corollary 12.
- **`lemma:certification`.** Replace the left inequality by Lemma 11, and update the paragraph
  after it ("if $\varepsilon_{\max}\le\bar\varepsilon$…").
- **§3.2.** Remove "Consistently with the well-posedness of the projector, $\tilde A_*$ has
  full row rank". $T(x_*)=\operatorname{null}\tilde A_*$ always has an orthonormal basis.
- **Appendix A.** Add the safeguard to Algorithm `algo:cauchy_point`.
- **Bibliography.** Conn–Gould–Toint 1988 (Theorem 11) and the TR book §12.2 are already
  there. Add Haraux 1977 if the critical cone is mentioned.

### 7.2 Code (`cauchy_step!`, `SubspaceProjector` method)

The box method (`CoordinateSubspaceProjector`) needs no change, by Proposition 4. In the
polyhedral loop:

```
L ← 0 ; crossed ← false ; best ← none                 # L = -gᵀs, crossed ⟺ t_G ≤ t
for each piece
    ...                                              # unchanged: stationary, parking, events
    if not stationary
        Hd ← H d ; φ1 ← gᵀd + sᵀHd ; φ2 ← dᵀHd ; ℓ ← −gᵀd
        τ_stop ← 0 if φ1 ≥ 0 ; −φ1/φ2 if φ2 > 0 and −φ1/φ2 < Δ ; none otherwise
        crossed ← crossed or max_{0<τ≤min(τ_stop, Δ)} [ pred + κ_G L + (φ1 + κ_G ℓ)τ + φ2τ²/2 ] ≥ 0
        if τ_stop ≠ none
            record (t+τ_stop, s+τ_stop·d, value, working set) in best if better
            crossed ⇒ break                           # usual case: unchanged result
            # premature: continue (the rest of the piece also updates crossed)
    end of path ⇒ record and break
    L += ℓΔ ; pred += φ1Δ + φ2Δ²/2 ; ...              # move to the breakpoint, unchanged
end
return best   # restore its working set in proj_op if it is not the last point
```

A small `eps_slope`-style tolerance on the crossing test is enough. The example of §3.2 makes
a good regression test: the decrease must stay above ~0.2 as δ → 0.

### 7.3 Numerical evidence

Random polyhedra ($(n,m)\in\{(10,3),(40,8)\}$, $A$ Gaussian, box $[-1,1]$, $x=P_\Omega(z)$
with $z\sim3\mathcal N(0,I)$, so many bounds are active), $g\sim3\mathcal N(0,I)$, $H=J^\top J$ with $J$ of scale 0.1, 1 or 3, and
$\Delta\in\{0.1,0.5,2\}$. The code examines a median of 2 pieces (at most 35) up to $t_1$.
Over 299 paths:

| $\kappa_G$ | premature $t_1$ | worst $q(s^C_{\rm code})/q(s^C_{\rm safe})$ | extra pieces (median / max) |
|---|---|---|---|
| 0.60 | 62 | 0.993 | 2 / 14 |
| 0.75 | 53 | 0.993 | 1 / 14 |
| 0.90 | 33 | 0.997 | 1 / 7 |
| 0.99 | 8 | 1.000 | 1 / 5 |
| 0.999 | 0 | 1.000 | — |

- **On generic problems the first local minimizer is right.** The safeguard almost never
  improves the step: it only buys a guarantee against the adversarial geometry of §3.2.
- **Choice of $\kappa_G$.** A value close to 1, say 0.99, keeps the extra cost negligible. The
  theoretical constant scales like $\kappa_G(1-\kappa_G)$. Any $\kappa_G\ge\frac12$ also
  accepts a minimizer inside the first piece.
- **Checks on the same instances.**
  - Theorem 7 held with a margin of at least 4.1. The $\chi$ version,
    $\frac{\kappa_G\chi}2\min(\Delta,1,4(1-\kappa_G)\chi/(n(1+\|H\|)))$, held with a margin of
    at least 2.4.
  - $\|s\|^2\le tL$ held to rounding.
  - $L$ was nondecreasing.

---

## 8. Remarks

1. **Why not keep the pure first local minimizer?** Its guarantee needs the speed not to
   increase after $t_1$, which only the future of the path can tell (§3.1). A safeguard has
   to look ahead somehow. The Goldstein crossing is the cheapest look-ahead with a proof, and
   it is already computable from the quantities of the scan.
2. **Stationary pieces** are the limit $a=0$ of §3.2. With the safeguard, skipping them is
   still the right choice: $\psi$ and $L$ are constant there, so they never trigger the
   crossing.
3. **Number of pieces.** The safeguard can only lengthen the scan up to $t^\sharp$, and still
   finitely (Proposition 1). The worst case is still the number of faces.
4. **What is not covered.** Rounding tolerances, parking and the `maxpieces` cap.
   - On the cap, the current $s$ still satisfies (C) if the crossing has been seen; otherwise
     the guarantee is lost, as before.
   - Failure of the local cone projection only affects the direction, not the path theory,
     since $d$ stays a feasible descent direction.

---

## Appendix — reproducing the counterexample

With the helpers of `test/cauchy_path.jl` (`cauchy_path`, `proj_dykstra`):

```julia
A = [1.0 1.0 1.0 0.0]
xlow, xupp = [-1.0, -Inf, -Inf, -Inf], [Inf, 1.0, Inf, Inf]
δ = 1e-4; a = δ / 2
x = [-1.0, 1.0, 0.0, 0.0] + δ * [1.0, -3.0, 2.0, 0.0]
g = [0.3, -0.8, 0.5, a]
v = [0.0, -1.0, 0.0, 1.0]
res = cauchy_path(A, xlow, xupp, x, g, 1.0; J = reshape(v, 1, 4))
res.pred                                                # ≈ -3.7e-4
norm(proj_dykstra(A, xlow .- x, xupp .- x, -g))         # h(x) ≈ 0.1414
```
