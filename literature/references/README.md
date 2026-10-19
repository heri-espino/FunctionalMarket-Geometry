# Literature map — verified metadata / online abstracts (2026-10-18)

**Source-status caveat:** Bibliographic titles, dates, DOI and the stated scope were checked using scholarly/publisher pages. **Full-text PDFs were not retrieved or page-by-page inspected.** `pdf/` and `markdown/` remain empty by design (README files only). Do not claim a source has been critically reviewed from its full text yet.

| Work | Verified link | Why relevant / limitation |
|---|---|---|
| Saart, P. W. & Xia, Y. (2022), *Functional time series approach to analysing asset returns co-movements*, Journal of Econometrics 229(1), 127–151 | [DOI:10.1016/j.jeconom.2020.11.012](https://doi.org/10.1016/j.jeconom.2020.11.012), [Cardiff record](https://orca.cardiff.ac.uk/id/eprint/139818/) | Functional PCA and evolving co-movements already exist. Strongest novelty challenge. |
| Isogai, T. (2017), *Dynamic correlation network analysis of financial asset returns with network clustering*, Applied Network Science 2 | [DOI:10.1007/s41109-017-0031-6](https://doi.org/10.1007/s41109-017-0031-6) | Dynamic financial networks and temporal market classifications are established. |
| Isogai, T. (2016), *Building a dynamic correlation network for fat-tailed financial asset returns*, Applied Network Science 1, article 7 | [DOI:10.1007/s41109-016-0008-x](https://doi.org/10.1007/s41109-016-0008-x) | Rolling Pearson estimates can be misleading under fat tails; volatility-filtered comparisons needed. |
| Ferreira et al. (2021), *Dynamic Connectivity in a Financial Network Using Time-Varying DCCA Correlation Coefficients*, Econometric Research in Finance | [DOI:10.2478/erfin-2021-0004](https://doi.org/10.2478/erfin-2021-0004) | Alternative time-varying network dependence estimators. |
| *Explaining Dynamic Changes in Various Asset’s Relationships in Financial Markets* (2021/2022), Review of Socionetwork Strategies | [Publisher](https://link.springer.com/article/10.1007/s12626-021-00094-5) | Demonstrates pre-existing standardized window DTW-based financial networks. |
| *A Temporal Graph Learning Framework for Lead-Lag Detection in Financial Markets* (2026), Machine Learning | [Publisher](https://link.springer.com/article/10.1007/s10994-026-07113-y) | Directed lead/lag graph comparisons exist; separate from symmetric functional distances. |

## Reading plan

1. Download legally accessible author/publisher PDFs into `literature/pdf/` **only if licensing permits local storage**.
2. Extract markdown with a provenance log into `literature/markdown/`.
3. Record exact definitions, closest baseline, assumptions and page references here (or in one `.md` per paper).
4. Update `paper/RESEARCH_DIRECTION.md` with what the new model does *not* reproduce.

**Novelty cannot be decided from titles and abstracts alone.**

## Guerrero-style trend estimation (added 2026-10-18)

| Work | Source | Relevance |
|---|---|---|
| Guerrero, V. M. (2007). *Time series smoothing by penalized least squares*. Statistics & Probability Letters 77(12), 1225–1234 | [DOI:10.1016/j.spl.2007.03.006](https://doi.org/10.1016/j.spl.2007.03.006) | Mean-centered difference penalty, plug-in mean and smoothness index. We use normalized S from Trend-Estimation. Publisher metadata/abstract checked; not a stored licensed PDF. |
| Espino et al. (work-in-progress). *Trend-Estimation* software and research notes | [Research repository](https://github.com/heri-espino/Trend-Estimation), especially `notes/conceptos/06-Media-de-diferencias-Guerrero.md` and `working_papers/THEORETICAL_CONTRIBUTIONS.md` | Original project specifies forecasting selectors, endpoints and distinction between normalized S and Guerrero plug-in mu. Not externally peer reviewed; do not imply its proposed selection methods are already established findings. |

## EXP-003: clarifying variance-ratio maximum likelihood

The Guerrero 2007 *time series smoothing* paper supports the Gaussian innovation/observation model with variance ratio `lambda = sigma_eta^2 / sigma_eps^2`. Its explicitly advertised *smoothing-parameter selection tool* is a **controlled smoothness index**, not a verified original maximum-likelihood estimator. The experiment's Gaussian profile ML of `D_d y` is an independently derived comparator motivated by the stochastic model. The manuscript must not cite Guerrero (2007) as introducing this lambda MLE without reading and verifying full text. See [EXP-003 notes](../../market_geometry/notes/003-order_lambda_loocv/README.md). The 2017 paper by Cortés-Toto, Guerrero and Reyes [DOI 10.1080/03610918.2015.1005236](https://doi.org/10.1080/03610918.2015.1005236) compares CV/GCV and additional parameter-selection criteria.

## EXP-004 smoothing-family comparators (citation-only, no copyrighted PDFs bundled)

- Eilers, P.H.C., & Marx, B.D. (1996), *Flexible smoothing with B-splines and penalties*, Statistical Science 11(2):89–121, https://doi.org/10.1214/ss/1038425655. Defines difference-penalty P-splines.
- Tibshirani, R.J. (2014), *Adaptive piecewise polynomial estimation via trend filtering*, Annals of Statistics 42(1):285–323, https://doi.org/10.1214/13-AOS1189. Motivates the unimplemented locally adaptive l1 branch.
- Guerrero, V.M. (2007), *Time series smoothing by penalized least squares*, Statistics & Probability Letters 77:1225–1234, https://doi.org/10.1016/j.spl.2007.03.006. Guerrero's smoothness index is not by itself Gaussian MLE selection.


## EXP-006 — window-local fPCA and graph-spectral precedents

- *Analysis of non-stationary dynamics in the financial system* (2014),
  Economics Letters, DOI [10.1016/j.econlet.2013.09.026](https://doi.org/10.1016/j.econlet.2013.09.026):
  moving-window functional PCA for financial instability. Abstract/publisher
  summary consulted; not a claim to have read or saved the full manuscript.
- *Dynamical analysis of financial stocks network: Improving forecasting using
  network properties* (2025), PLOS ONE,
  [DOI 10.1371/journal.pone.0319985](https://doi.org/10.1371/journal.pone.0319985):
  temporal network statistics and stock forecasting. Open article available.
- Nobi, A., and Lee, J. W. (2016), *State and group dynamics of world stock
  market by principal component analysis*, Physica A 450, 85–94,
  [RePEc bibliographic record](https://ideas.repec.org/a/eee/phsmap/v450y2016icp85-94.html):
  correlation-matrix PCs and market structure. Distinguish this from FPCA.

References are bibliographic links only. No PDFs or copyrighted texts are bundled.
