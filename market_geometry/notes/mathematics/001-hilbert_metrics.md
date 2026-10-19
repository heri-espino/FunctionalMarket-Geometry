# Mathematical note 001 — Hilbert metrics for return trajectories

## Definitions

Fix $n$ assets and a trailing window of $W\ge 2$ aligned return observations. Let $z_i\in\mathbb R^W$ have empirical mean zero and **population-normalized** variance $W^{-1}\sum_t z_i(t)^2=1$. Let $\widehat\rho_{ij}=W^{-1}\sum_t z_i(t)z_j(t)$ be within-window Pearson correlation.

The real Hilbert space $H^1([0,1])$ contains square-integrable functions whose first *weak derivative* is also square-integrable. For fixed $\lambda>0$, a weighted inner product is

$$
\langle f,g\rangle_{H^1_\lambda}
=\int_0^1f(u)g(u)du+\lambda\int_0^1 f'(u)g'(u)du.
$$

The derivative-only term is a **seminorm** (constant offsets vanish). The sum is an inner product and induces a norm; for $\lambda=0$ it reduces to the $L^2$ norm. On B-spline fitted functions, $d_\lambda(f,g)=\|f-g\|_{H^1_\lambda}$ is a metric, while on raw observations after smoothing/projection it is generally a pseudometric.

## Proposition 1 — exact sampled L2/Pearson equivalence

For such standardized return vectors,

$$
\frac1W\|z_i-z_j\|_2^2
=\frac1W\left(\|z_i\|^2+\|z_j\|^2-2\langle z_i,z_j\rangle\right)
=1+1-2\widehat\rho_{ij}=2(1-\widehat\rho_{ij}).
$$

Thus the raw sampled $L^2$ distance $d=\sqrt{2(1-\widehat\rho)}$ is *exactly* a monotone transformation of Pearson correlation. No independent geometric information is generated. The proposition does not imply the same identity for heavily smoothed spline fits or for derivative-weighted norms.

## Proposition 2 — Gram matrix representation

Let $f_i(u)=\sum_{k=1}^K\beta_{ik}B_k(u)$. Let $G_0$ and $G_1$ be basis Gram matrices:

$$
(G_0)_{jk}=\int_0^1 B_j(u)B_k(u)du,\qquad
(G_1)_{jk}=\int_0^1 \frac{B_j'(u)B_k'(u)}{(W-1)^2}du.
$$

Let $G_\lambda=G_0+\lambda G_1$. With $\delta=\beta_i-\beta_j$,

$$
d_\lambda^2(i,j)=\delta^{\mathsf T}G_\lambda\delta\ge0.
$$

The nonnegativity follows from $x^{\mathsf T}G_\lambda x=\int (\sum_kx_kB_k)^2+\lambda\int(\sum_kx_k B_k'/(W-1))^2$. A linearly independent spline basis yields positive-definite $G_\lambda$ in coefficient space (with numerical quadrature error in the implementation). Triangle inequality follows from the norm on the fitted-function space.

**Units:** B-splines are constructed with $u\in[0,1]$, while the process takes one step per trading day; hence $df/dt=(W-1)^{-1}df/du$. Without this factor, tuning $W$ changes the derivative scale mechanically.

## Proposition 3 — the first-derivative term is a lagged-covariance statistic

Consider standardized, jointly weakly stationary returns $X_{i,t},X_{j,t}$, each with mean zero and variance 1. Write $Y_t=X_{i,t}-X_{j,t}$ and $C_Y(h)=\operatorname{Cov}(Y_t,Y_{t-h})$. Then

$$
\mathbb E[(Y_t-Y_{t-1})^2]
=\mathbb E[Y_t^2]+\mathbb E[Y_{t-1}^2]-2\mathbb E[Y_tY_{t-1}]
=2[C_Y(0)-C_Y(1)].
$$

Since $C_Y(0)=2(1-\rho_{ij}(0))$, and defining $\gamma_{ab}(1)=\operatorname{Cov}(X_{a,t},X_{b,t-1})$,

$$
C_Y(1)=\gamma_{ii}(1)+\gamma_{jj}(1)-\gamma_{ij}(1)-\gamma_{ji}(1).
$$

Therefore the expected discrete derivative penalty contains **nothing that is absent from contemporaneous and lag-one second-order covariance**. The penalized-spline derivative norm is a smoothed, frequency-selective version of this idea, not a fundamentally new type of statistical dependence. A meaningful gain must come from regularization, finite-sample stability, a particular task, or genuinely richer operators—not from renaming the same covariances as a Hilbert metric.

In EXP-001, with distinct members of the **same group**, $Y_t=\sqrt{1-a}(\epsilon_{i,t}-\epsilon_{j,t})$, hence

$$
\mathbb E[(\Delta Y_t)^2\mid g(i)=g(j),i\ne j]=4(1-a).
$$

For **different groups**, $C_Y(0)=2$ and $C_Y(1)=2a\phi_s$, so

$$
\mathbb E[(\Delta Y_t)^2\mid g(i)\ne g(j),s]
=4(1-a\phi_s).
$$

With $a=0.70$, $\phi_0=0.82$ and $\phi_1=-0.65$, the between-group expectation changes from $1.704$ to $5.82$, while the within-group expectation remains $1.20$. This **analytical contrast**, not a claim of unknown nonlinear financial dynamics, motivates the pilot.

## Important distinction: three different concepts

1. **Geometric proximity** — two fitted return paths are close under a chosen norm.
2. **Statistical dependence** — a property of a joint law; similar paths in a finite sample are not a dependence estimator by definition.
3. **Causality / lead-lag predictability** — directed, model- and assumption-dependent; an undirected $H^1$ distance cannot establish it.

## Testable methodological predictions

- Under a mechanism where time-autocorrelation changes but contemporaneous covariance does not, an H1 derivative statistic may shift; **so should ordinary lag-aware baselines**.
- With $\lambda=0$, spline smoothing alone can change Pearson-equivalent distances but must be treated as denoising, not a fundamentally new Hilbert geometry.
- With high observation noise, derivative terms may become less stable; evaluate across $\gamma$, $\lambda$, window size and sampling frequency.
- If empirically a metric is learned from event labels, report the labels and selection procedure and evaluate on unseen future events.
