# Variational Reconstruction {#variational_reconstruction}


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Variational Reconstruction](#variational-reconstruction)
  - [Functional Design:](#functional-design)
    - [Selecting Inner Product](#selecting-inner-product)
    - [Other Forms](#other-forms)
  - [Reconstruction Matrices](#reconstruction-matrices)

<!-- /code_chunk_output -->



**[PDF version of this page](Variational_Reconstruction.pdf)**


Reconstruction from mean values with piecewise zero-mean basis:

$$
u_i(\mathbf{x})
=\overline{u}_i
+\sum_{l=1}^{N_{base}}{
    u^l_i\varphi^l_i(\mathbf{x})}
$$


## Functional Design:

For face $f$ with cell on both sides $L,R$, reconstruction functional on the face is designed as:

$$
I_f=w_g(f)\int_f{d\Gamma
    \sum_{p=0}^k
    {w_d(p)^2
        \left\|
            \mathcal{D}_pu_L
            -\mathcal{D}_pu_R
    \right\|_{\langle\rangle _{f,p}}^2}}
$$
where 
- $w_g$ is Geometric Weight, which could well be dimensioned. An original choice is to use $w_g=S_f^{-1}$, where $S_f$ is area of $f$. 
- $w_d$ is Derivative Weight, which is defined to be *dimensionless* here.
- $\mathcal{D}_p u$ is a tensor-like (covariance only in linear transformation of space from
, no curvilinear) variable representing the derivatives of $u$ , for
example, the Cartesian components:
$$
[\mathcal{D}_3 u]_{ijk}
\equiv
[u]_{x_jx_jx_k}
\equiv
\frac{\partial^3u}{
    \partial x_i \partial x_j\partial x_k}
$$
- $\left\|\cdot\right\|_{\langle\rangle _{f,p}}^2$ is defined with 
$$
\left\|A\right\|_{\langle\rangle _{f,p}}^2
=\langle A,A \rangle _{f,p}
$$
where $A$ is a tensor of p_th order. 
The $\langle,\rangle$ is a inner product.

### Selecting Inner Product

The Normal Functional from Wang:

$$
\langle \mathcal{D}_3 u,\mathcal{D}_3 v\rangle _{f,3}
=d_f^6 \partial_{nnn}u \partial_{nnn}v
$$

X-Y aligned from Pan:

$$
\begin{aligned}
\langle \mathcal{D}_3 u,\mathcal{D}_3 v\rangle _{f,3}
&=(\Delta_x^3 \partial_{xxx} u)
(\Delta_x^3 \partial_{xxx} v)\\
&+(3\Delta_x^2\Delta_y \partial_{xxy} u)
(3\Delta_x^2\Delta_y \partial_{xxy} v)\\
&+(3\Delta_x\Delta_y^2 \partial_{xyy} u)
(3\Delta_x\Delta_y^2 \partial_{xyy} v)\\
&+(\Delta_y^3 \partial_{yyy} u)
(\Delta_y^3 \partial_{yyy} v)\\
\end{aligned}
$$

Pre-Isotropic from Huang:

$$
\begin{aligned}
\langle \mathcal{D}_3 u,\mathcal{D}_3 v\rangle _{f,3}
&=(d_f^3 \partial_{xxx} u)
(d_f^3 \partial_{xxx} v)\\
&+(d_f^3 \partial_{xxy} u)
(d_f^3 \partial_{xxy} v)\\
&+(d_f^3 \partial_{xyy} u)
(d_f^3 \partial_{xyy} v)\\
&+(d_f^3 \partial_{yyy} u)
(d_f^3 \partial_{yyy} v)\\
\end{aligned}
$$

### Other Forms

Could be re-written to:

$$
I_f=w_g(f)\int_f{d\Gamma
    \sum_{p=0}^k
    {
        w_d(p)^2
        L_f^{2p}
        \left\|
            \mathcal{D}_pu_L
            -\mathcal{D}_pu_R
        \right\|_{\langle\rangle _{f,p}}^2
    }
}
$$

- $L_f$ is a facial length reference, all the anisotropy within derivative components are dealt in the inner product with a relative (dimensionless form). *Out put of inner product becomes dimensional*.


## Reconstruction Matrices

$$
\begin{aligned} 
        A_{mn}^i     & = \sum_{j\in S_i} {
        w_g(f_{i-j}) \int_{f_{i-j}}{\sum_{p=0}^k{
        w_d(p)^2 \left\langle{\mathcal{D}_p(\varphi^m_i),\mathcal{D}_p(\varphi^n_i)}\right\rangle_{f_{i-j},p}
        } d\Gamma}}                        \\
        B_{mn}^{i-j} & = {
        w_g(f_{i-j}) \int_{f_{i-j}}{\sum_{p=0}^k{
        w_d(p)^2 \left\langle{\mathcal{D}_p(\varphi^m_i),\mathcal{D}_p(\varphi^n_j)}\right\rangle_{f_{i-j},p}
        } d\Gamma}}                        \\
        b_{m}^{i-j}  & = {
        w_g(f_{i-j}) \int_{f_{i-j}}{{
        w_d(0)^2 \left\langle{\varphi^m_i,1}\right\rangle_{f_{i-j},p}
        } d\Gamma}}
    \end{aligned}
$$

with the reconstruction system in local form:

$$
A_{mn}^i u_i^n = \sum_{j\in S_i}{
        \left(B_{mn}^{i-j} u_j^n
        +b_{m}^{i-j}(\overline{u}_j - \overline{u}_i)
        \right)}
$$

