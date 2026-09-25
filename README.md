# GlobalFit-EGC

A Python implementation of **GlobalFit** for detecting and removing
**energy-generating cycles** (EGCs) in genome-scale metabolic models, following
[Fritzemeier et al. (2017)](https://doi.org/10.1371/journal.pcbi.1005494).
Built on COBRApy and the open-source HiGHS solver, so no commercial license is
needed.

## The problem

An EGC is a set of reactions that lets a metabolic model charge ATP (or another
energy metabolite) **without taking up any nutrients**. That is
thermodynamically impossible, but flux balance analysis (FBA) ignores
thermodynamics: it uses the free energy, and predicted growth gets inflated.
Fritzemeier et al. found EGCs in 68% of 350 published models, and in over 85%
of the automatically generated ones.

![Growth of iJO1366: curated, with EGCs, and corrected](docs/crecimiento_iJO1366.png)

In *E. coli* iJO1366, re-enabling the six reactions its authors blocked creates
EGCs that inflate growth by 41%. GlobalFit-EGC finds the minimal fix (a single
change) and brings growth back to its real value.

## Validation against the paper

Table S2 of the paper lists what GlobalFit detected and removed in each model.
Results of this implementation on models downloaded from BiGG today:

| Model | EGCs detected (normalized) | Paper's removal | GlobalFit-EGC |
|---|---|---|---|
| iND750 (*S. cerevisiae*) | ACCOA 1 — **same** | `ACOAH` forward | `ACOAH` forward |
| iJN746 (*P. putida*) | ATP 1, CTP 1, GTP 1, UTP 1, ACCOA 1, PROTON 4 — **same** | `ACALD` forward | `ACALD` forward or `ALDD2x_copy2` backward |
| iJN746, ATP synthase kept | | `ALDD2x_copy2` backward | `ALDD2x_copy2` backward or `ACALD` forward |
| iJO1366 + 6 reactions | ATP, CTP, GTP, UTP, ITP, ACCOA, PROTON | `MOX` backward or `SPODM` forward (Fig. 6C) | `MOX` backward, `SPODM` forward or `MDH` forward |
| iRC1080 (*C. reinhardtii*) | GTP 1, UTP 1, ITP 1, ACCOA 2.25 — **same** | 49 changes | 49 changes, 41 shared with the paper |

Every solution is checked with an independent FBA: the corrected model generates
no energy without nutrients and can still grow. Runs take from under a second
to a few minutes on a laptop (iRC1080 is the slowest); the paper used CPLEX on
an 8-core server.

In iRC1080 the optimal fix has the same size as the paper's (49 changes), and
the differences are equal-cost alternatives, mostly other acyltransferases from
the same lipid family. Applied to today's BiGG version, the paper's own set
leaves one ITP cycle open, because one of its reactions (`ITPA`) no longer
exists in the model.

To reproduce: `python examples/validate_paper.py`.

## Automatically generated models

The paper's central finding is that automatic reconstructions are riddled with
EGCs. Models of the same two organisms built with different tools (from the
[merlin v4 benchmark](https://github.com/BioSystemsUM/merlinv4_paper)):

| | *L. plantarum* | *B. pertussis* |
|---|---|---|
| **ModelSEED** (automatic) | EGCs for 6 energy metabolites | EGCs for 12 |
| **CarveMe** (automatic) | none | EGCs for all 15 |
| **Manually curated** | none | none |

GlobalFit-EGC removes all of them in seconds. In *B. pertussis* (ModelSEED) it
finds the exact cycle of the paper's Fig. 6D: `rxn00379` makes APS from ATP and
sulfate, and `rxn09240` runs the reverse reaction while also charging a GTP.
Removing either one breaks it. The notebook walks through these cases.

## Installation

```bash
git clone https://github.com/benjamitchell/GlobalFit-EGC.git
cd GlobalFit-EGC
pip install -e ".[dev,demo]"
pytest
```

Requires Python ≥ 3.10.

## Usage

```python
from globalfit import (
    add_energy_dissipation_reactions, detect_egcs, evidence_weights,
    globalfit, load_bigg_model, verify,
)

model = load_bigg_model("iJO1366")               # downloads from BiGG to ./modelos
edrs = add_energy_dissipation_reactions(model)   # the 15 EDRs of Table S1

detect_egcs(model, edrs)                          # {"EDR_ATP": 0.0, ...}; > 0 means an EGC

results = globalfit(
    model,
    energy_rxns=edrs,
    min_growth=0.1,                   # minimal growth after the fix
    weights=evidence_weights(model),  # optional: break ties by evidence
    protected=["ATPS4rpp"],           # optional: never remove these
    n_solutions=3,                    # enumerate alternatives
)
for r in results:
    print(r.objective, [str(x) for x in r.removals])
    energy, growth = verify(model, r.removals, edrs)
```

For ModelSEED models, pass `MODELSEED_DISSIPATION_REACTIONS` to
`add_energy_dissipation_reactions`. The notebook
[`notebooks/demo_globalfit.ipynb`](notebooks/demo_globalfit.ipynb) runs the full
pipeline with results (the notebook is in Spanish).

## How it works

### Detection

Each model gets an **energy dissipation reaction** (EDR) for 15 energy
metabolites (ATP, CTP, GTP, UTP, ITP, NADH, NADPH, FADH₂, FMNH₂, ubiquinol-8,
menaquinol-8, 2-demethylmenaquinol-8, acetyl-CoA, glutamate, and the proton
gradient), e.g. `ATP + H₂O → ADP + Pi + H⁺`. Nutrient uptake is blocked and each
EDR is maximized: any positive flux reveals an EGC. The EDRs for redox cofactors
are **intentionally charge-unbalanced**: including the electron acceptor would
let the energy be recycled inside the network, and the EDR could carry flux
even without an EGC.

### Correction: the bilevel problem

GlobalFit looks for the fewest reaction **directions** to remove (binaries
$\delta^F_j, \delta^B_j$) such that two conditions hold at once:

$$
\begin{aligned}
\min_{\delta}\quad & \textstyle\sum_j w^F_j\,\delta^F_j + w^B_j\,\delta^B_j \\
\text{s.t.}\quad & S\,v = 0,\quad lb_j(1-\delta^B_j) \le v_j \le ub_j(1-\delta^F_j),\quad v_{\text{bio}} \ge T
&& \text{(growth)}\\
& \max_{w}\ \Big\lbrace \textstyle\sum_{d \in \text{EDR}} w_d \;:\; S\,w = 0,\; l_j(1-\delta^B_j) \le w_j \le u_j(1-\delta^F_j)\Big\rbrace = 0
&& \text{(no nutrients, no EGCs)}
\end{aligned}
$$

The second condition is an optimization problem nested inside another one.

### From bilevel to a single MILP, via duality

The inner problem is an LP that always admits $w = 0$, so its optimum is
$\ge 0$. By LP duality, the optimum is $\le 0$ **if and only if** there is a
dual certificate $(\lambda, \alpha \ge 0, \beta \ge 0)$ with

$$
S^\top \lambda + \alpha - \beta = c, \qquad
\sum_j u_j(1-\delta^F_j)\,\alpha_j + |l_j|(1-\delta^B_j)\,\beta_j = 0,
$$

where $c$ is 1 on the EDRs and 0 elsewhere. Every term of the sum is
non-negative, so the sum is zero only if $\alpha_j = 0$ for every forward
direction that is still allowed (likewise for $\beta$). That linearizes as

$$
\alpha_j \le M\,\delta^F_j, \qquad \beta_j \le M\,\delta^B_j .
$$

The result is a single MILP with no flux variables for the no-growth case,
solved by HiGHS through `scipy.optimize.milp`.

**Interpretation.** $\lambda$ acts as a chemical potential for each metabolite:
every reaction that may run forward must go "downhill"
($S_j^\top\lambda \ge c_j$), and every reaction that may run backward, "uphill".
An EGC exists exactly when no set of potentials is consistent with all allowed
directions.

**Robustness.** By weak duality, any feasible solution is a valid certificate.
A Big-M that is too small can only yield fixes with more changes than
necessary, never a model that still has EGCs.

**Verification.** MILP solvers accept binaries within a tolerance, so a
"removed" direction with $\delta = 1 - 10^{-6}$ can still carry
$ub \cdot 10^{-6}$ of flux. With dozens of removals in a large model, that leak
can fake growth (this happened on iRC1080). `globalfit` therefore rounds each
candidate and re-checks it with exact LPs, growth and absence of EGCs, before
returning it; spurious candidates are cut off and the MILP is solved again.

### Weights and alternatives

- `weights` sets a cost for each direction. `evidence_weights` implements the
  two criteria the paper suggests: removing reactions without a gene rule (GPR)
  is cheaper, and removing an irreversible reaction entirely is more expensive
  than removing one direction of a reversible one. In iJO1366 this breaks the
  tie in favor of `MOX`, the only one of the three solutions with no gene.
- `n_solutions` enumerates alternatives with *no-good* cuts. Each cut forbids a
  solution and its supersets, so alternatives come out minimal and in order of
  cost.

## Differences from the paper

| | Fritzemeier et al. (2017) | GlobalFit-EGC |
|---|---|---|
| Language / solver | R (sybil) + CPLEX | Python (COBRApy) + HiGHS |
| Bilevel reformulation | following Hartleb et al. (2016) | strong duality (see above) |
| Weights | supported; uniform weights used | uniform by default; optional `evidence_weights` |
| EDR namespaces | BiGG, ModelSEED, MetaNetX | BiGG and ModelSEED (others can be passed) |

## Limitations

- Predefined EDRs exist for BiGG and ModelSEED ids. For MetaNetX or other
  namespaces, pass your own dictionary to `add_energy_dissipation_reactions`.
- The paper does not state which biomass reaction it used for iRC1080 (the
  SBML has none set); all three give 49-change fixes.
- `cobra.io.load_model` no longer downloads from BiGG (it uses `http://` and the
  server redirects to `https://`), so the package ships `load_bigg_model`.

## Repository layout

```
globalfit/            the package (core.py: detection, GlobalFit, weights; io.py: BiGG)
tests/                tests on the E. coli core model with injected EGCs
examples/             run_iJO1366.py and validate_paper.py
notebooks/            demo_globalfit.ipynb (Spanish)
  original_2025/      notebooks from the original course project
informe/              course report (Spanish), the paper and its supplementary data
docs/                 figures
```

## History

This repository grew out of a project for the course *Modelamiento y Análisis
de Redes Biológicas* (MA5405, Universidad de Chile, June 2025). The course
version, in [`notebooks/original_2025/`](notebooks/original_2025/), never
reached a feasible solution, for two reasons: to speed up testing it used 12 of
iJO1366's ~2,580 reactions while keeping strict mass balance, which forces
biomass to zero; and the inner problem was written as $\min v_{ATP}$, which is
always 0, instead of $\max$. This version is a from-scratch rewrite on the full
model.

## Authors

- **Benjamín Mitchell García** ([@benjamitchell](https://github.com/benjamitchell))
- **Millaray Díaz Araujo** ([@Millaray-DA](https://github.com/Millaray-DA))

Course project for MA5405, taught by Vicente Acuña, Alejandro Maass and
Sebastián Mendoza.

## License

Code under the [MIT license](LICENSE). The Fritzemeier et al. paper and its
supplementary data in `informe/` are distributed under CC BY 4.0.

## References

- Fritzemeier CJ, Hartleb D, Szappanos B, Papp B, Lercher MJ (2017). *Erroneous
  energy-generating cycles in published genome scale metabolic networks:
  Identification and removal.* PLoS Comput Biol 13(4): e1005494.
  https://doi.org/10.1371/journal.pcbi.1005494
- Hartleb D, Jarre F, Lercher MJ (2016). *Improved Metabolic Models for E. coli
  and Mycoplasma genitalium from GlobalFit, an Algorithm That Simultaneously
  Matches Growth and Non-Growth Data Sets.* PLoS Comput Biol 12(8): e1005036.
  https://doi.org/10.1371/journal.pcbi.1005036
- Benchmark models used for the ModelSEED/CarveMe examples, from the merlin v4
  paper repository (MIT license): https://github.com/BioSystemsUM/merlinv4_paper
