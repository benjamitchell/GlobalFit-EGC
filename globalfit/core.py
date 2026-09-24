"""GlobalFit para eliminar ciclos generadores de energía (EGCs).

Formulación (Fritzemeier et al., 2017, adaptada):

    min  sum_j  delta^F_j + delta^B_j
    s.a. (crecimiento)
         S v = 0
         lb_j (1 - delta^B_j) <= v_j <= ub_j (1 - delta^F_j)
         v_bio >= T
         (sin nutrientes, sin EGCs)
         max { v_ATP : S w = 0, l_j (1 - delta^B_j) <= w_j <= u_j (1 - delta^F_j) } = 0

El problema interno es un LP que siempre admite w = 0, así que su óptimo es
>= 0. Por dualidad de LP, el óptimo es <= 0 si y solo si existen
(lambda, alpha >= 0, beta >= 0) tales que

    S^T lambda + alpha - beta = e_ATP
    sum_j  u_j (1 - delta^F_j) alpha_j  +  |l_j| (1 - delta^B_j) beta_j  = 0

Como cada término es no negativo, la segunda condición equivale a
alpha_j = 0 salvo que la dirección forward de j esté eliminada (idem beta con
backward), que se linealiza como alpha_j <= M delta^F_j. Esto reemplaza las
condiciones KKT con binarias de complementariedad: no hacen falta variables
de flujo del problema interno.

Interpretación: lambda funciona como un "potencial químico" de cada
metabolito. Si una reacción puede ir hacia adelante, S_j^T lambda >= c_j; si
puede ir hacia atrás, S_j^T lambda <= c_j. Un EGC existe justo cuando no hay
potenciales consistentes con todas las direcciones permitidas.

Cualquier solución factible es un certificado válido (dualidad débil): un
Big-M demasiado chico solo puede empeorar la optimalidad, nunca producir un
modelo que todavía tenga EGCs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from cobra import Model, Reaction
from cobra.util import create_stoichiometric_matrix
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, vstack

TOL = 1e-6
MAX_REJECTED = 20  # candidatos del MILP que fallan la verificación antes de rendirse


@dataclass
class Removal:
    """Una dirección de reacción que GlobalFit propone eliminar."""

    reaction: str
    direction: str  # "forward" o "backward"

    def __str__(self) -> str:
        return f"{self.reaction} ({self.direction})"


@dataclass
class GlobalFitResult:
    status: str
    removals: list[Removal] = field(default_factory=list)
    objective: float | None = None
    growth: float | None = None
    mip_gap: float | None = None


# Reacciones de disipación de energía (EDR) de Fritzemeier et al. (2017),
# tabla S1, traducidas a ids de BiGG. Como en el paper, las de cofactores
# redox no están balanceadas en carga: se omite el aceptor de electrones para
# que la energía no pueda reciclarse dentro de la red.
BIGG_DISSIPATION_REACTIONS: dict[str, dict[str, float]] = {
    "ATP": {"atp_c": -1, "h2o_c": -1, "adp_c": 1, "h_c": 1, "pi_c": 1},
    "CTP": {"ctp_c": -1, "h2o_c": -1, "cdp_c": 1, "h_c": 1, "pi_c": 1},
    "GTP": {"gtp_c": -1, "h2o_c": -1, "gdp_c": 1, "h_c": 1, "pi_c": 1},
    "UTP": {"utp_c": -1, "h2o_c": -1, "udp_c": 1, "h_c": 1, "pi_c": 1},
    "ITP": {"itp_c": -1, "h2o_c": -1, "idp_c": 1, "h_c": 1, "pi_c": 1},
    "NADH": {"nadh_c": -1, "h_c": 1, "nad_c": 1},
    "NADPH": {"nadph_c": -1, "h_c": 1, "nadp_c": 1},
    "FADH2": {"fadh2_c": -1, "h_c": 2, "fad_c": 1},
    "FMNH2": {"fmnh2_c": -1, "h_c": 2, "fmn_c": 1},
    "Q8H2": {"q8h2_c": -1, "h_c": 2, "q8_c": 1},
    "MQL8": {"mql8_c": -1, "h_c": 2, "mqn8_c": 1},
    "DMMQL8": {"2dmmql8_c": -1, "h_c": 2, "2dmmq8_c": 1},
    "ACCOA": {"accoa_c": -1, "h2o_c": -1, "h_c": 1, "ac_c": 1, "coa_c": 1},
    "GLU": {"glu__L_c": -1, "h2o_c": -1, "akg_c": 1, "nh4_c": 1, "h_c": 2},
    "PROTON": {"h_p": -1, "h_c": 1},
}


# Las mismas 15 EDR con ids de ModelSEED (compartimentos c0/p0), verificados
# contra ModelSEEDDatabase/Biochemistry/compounds.tsv. Los modelos de
# ModelSEED suelen no tener periplasmo, así que PROTON casi nunca aplica.
MODELSEED_DISSIPATION_REACTIONS: dict[str, dict[str, float]] = {
    "ATP": {"cpd00002_c0": -1, "cpd00001_c0": -1, "cpd00008_c0": 1, "cpd00067_c0": 1, "cpd00009_c0": 1},
    "CTP": {"cpd00052_c0": -1, "cpd00001_c0": -1, "cpd00096_c0": 1, "cpd00067_c0": 1, "cpd00009_c0": 1},
    "GTP": {"cpd00038_c0": -1, "cpd00001_c0": -1, "cpd00031_c0": 1, "cpd00067_c0": 1, "cpd00009_c0": 1},
    "UTP": {"cpd00062_c0": -1, "cpd00001_c0": -1, "cpd00014_c0": 1, "cpd00067_c0": 1, "cpd00009_c0": 1},
    "ITP": {"cpd00068_c0": -1, "cpd00001_c0": -1, "cpd00090_c0": 1, "cpd00067_c0": 1, "cpd00009_c0": 1},
    "NADH": {"cpd00004_c0": -1, "cpd00067_c0": 1, "cpd00003_c0": 1},
    "NADPH": {"cpd00005_c0": -1, "cpd00067_c0": 1, "cpd00006_c0": 1},
    "FADH2": {"cpd00982_c0": -1, "cpd00067_c0": 2, "cpd00015_c0": 1},
    "FMNH2": {"cpd01270_c0": -1, "cpd00067_c0": 2, "cpd00050_c0": 1},
    "Q8H2": {"cpd15561_c0": -1, "cpd00067_c0": 2, "cpd15560_c0": 1},
    "MQL8": {"cpd15499_c0": -1, "cpd00067_c0": 2, "cpd15500_c0": 1},
    "DMMQL8": {"cpd15353_c0": -1, "cpd00067_c0": 2, "cpd15352_c0": 1},
    "ACCOA": {"cpd00022_c0": -1, "cpd00001_c0": -1, "cpd00067_c0": 1, "cpd00029_c0": 1, "cpd00010_c0": 1},
    "GLU": {"cpd00023_c0": -1, "cpd00001_c0": -1, "cpd00024_c0": 1, "cpd00013_c0": 1, "cpd00067_c0": 2},
    "PROTON": {"cpd00067_p0": -1, "cpd00067_c0": 1},
}


def _as_list(energy_rxns: str | list[str]) -> list[str]:
    return [energy_rxns] if isinstance(energy_rxns, str) else list(energy_rxns)


def add_dissipation_reaction(
    model: Model, rxn_id: str, stoichiometry: dict[str, float], name: str = ""
) -> Reaction:
    """Agrega una reacción irreversible que disipa un metabolito energético.

    `stoichiometry` usa ids de metabolitos, p. ej. para NADH en BiGG:
    {"nadh_c": -1, "h_c": 1, "nad_c": 1}.
    """
    missing = [m for m in stoichiometry if m not in model.metabolites]
    if missing:
        raise KeyError(f"{rxn_id}: metabolitos ausentes en el modelo: {missing}")
    rxn = Reaction(rxn_id, name=name, lower_bound=0, upper_bound=1000)
    model.add_reactions([rxn])
    rxn.add_metabolites(
        {model.metabolites.get_by_id(m): coef for m, coef in stoichiometry.items()}
    )
    return rxn


def add_energy_dissipation_reactions(
    model: Model,
    reactions: dict[str, dict[str, float]] = BIGG_DISSIPATION_REACTIONS,
    prefix: str = "EDR_",
) -> list[str]:
    """Agrega las reacciones de disipación cuyos metabolitos existen todos en
    el modelo (las demás no podrían llevar flujo; el paper las descarta igual).

    Devuelve los ids de las reacciones agregadas, p. ej. "EDR_ATP".
    """
    added = []
    for name, stoichiometry in reactions.items():
        if all(m in model.metabolites for m in stoichiometry):
            add_dissipation_reaction(model, prefix + name, stoichiometry, name=f"{name} dissipation")
            added.append(prefix + name)
    return added


def detect_egcs(
    model: Model, energy_rxns: str | list[str] = "ATPM", normalize: bool = False
) -> dict[str, float]:
    """Máximo flujo de cada reacción de disipación de energía sin nutrientes.

    Un valor > 0 indica que el modelo tiene al menos un EGC para ese
    metabolito energético. Con `normalize=True`, como en el paper, el resto de
    las reacciones se acota a [-1, 1], de modo que cada EGC aporta según su
    rendimiento por vuelta y ciclos disjuntos se suman. El paper lo interpreta
    como una cota inferior del número de EGCs, pero es solo aproximado: un
    ciclo que carga 0,75 ATP por vuelta aporta 0,75.
    """
    energy_rxns = _as_list(energy_rxns)
    out = {}
    with model:
        # Mismas cotas que el problema interno de `globalfit`: nada forzado
        # (l <= 0 <= u) y sin captación de nutrientes.
        for rxn in model.reactions:
            lb, ub = min(rxn.lower_bound, 0), max(rxn.upper_bound, 0)
            if normalize and rxn.id not in energy_rxns:
                lb, ub = max(lb, -1), min(ub, 1)
            rxn.bounds = (lb, ub)
        for rxn in model.boundary:
            rxn.lower_bound = 0
        for rid in energy_rxns:
            model.objective = rid
            value = model.slim_optimize(error_value=np.nan)
            out[rid] = 0.0 if abs(value) < TOL else value
    return out


def apply_removals(model: Model, removals: list[Removal]) -> None:
    """Aplica las eliminaciones in place (usar dentro de `with model:`)."""
    for removal in removals:
        rxn = model.reactions.get_by_id(removal.reaction)
        if removal.direction == "forward":
            rxn.upper_bound = min(rxn.upper_bound, 0)
        else:
            rxn.lower_bound = max(rxn.lower_bound, 0)


def evidence_weights(
    model: Model, no_gene: float = 0.5, irreversible: float = 2.0
) -> dict[tuple[str, str], float]:
    """Pesos según los dos criterios que sugiere el paper (p. 6).

    - Una reacción sin regla génica (GPR) tiene poca evidencia genómica y es
      más barata de eliminar: su costo se multiplica por `no_gene`.
    - Es más probable que sobre una dirección de una reacción reversible que
      una reacción irreversible completa: eliminar la única dirección de una
      irreversible se multiplica por `irreversible`.
    """
    out = {}
    for rxn in model.reactions:
        base = 1.0 if rxn.gene_reaction_rule else no_gene
        if not rxn.reversibility:
            base *= irreversible
        out[(rxn.id, "forward")] = base
        out[(rxn.id, "backward")] = base
    return out


def _biomass_id(model: Model) -> str:
    ids = [r.id for r in model.reactions if r.objective_coefficient != 0]
    if len(ids) != 1:
        raise ValueError(
            f"Se esperaba una única reacción objetivo, hay {ids}; "
            "indica la de biomasa con `biomass_rxn=`"
        )
    return ids[0]


def globalfit(
    model: Model,
    energy_rxns: str | list[str] = "ATPM",
    biomass_rxn: str | None = None,
    min_growth: float = 0.1,
    candidates: list[str] | None = None,
    protected: list[str] = (),
    rich_medium: bool = False,
    weights: Mapping[str | tuple[str, str], float] | None = None,
    big_m: float = 1e3,
    time_limit: float = 600,
    n_solutions: int = 1,
    verbose: bool = False,
) -> list[GlobalFitResult]:
    """Busca el conjunto mínimo de direcciones a eliminar para quitar los EGCs.

    Parámetros
    ----------
    model : modelo COBRA; sus cotas actuales definen el medio de crecimiento.
    energy_rxns : reacción o lista de reacciones de disipación de energía
        (por defecto ATPM). Se exige que ninguna tenga flujo sin nutrientes;
        como todas son >= 0, basta con que el máximo de su suma sea 0, así
        que un solo bloque dual cubre todos los metabolitos energéticos.
    biomass_rxn : reacción de biomasa; por defecto, el objetivo del modelo.
    min_growth : crecimiento mínimo T exigido tras las eliminaciones.
    candidates : reacciones que se pueden modificar. Por defecto todas menos
        las de borde, la de biomasa y la de energía.
    protected : reacciones que nunca se eliminan (p. ej. la ATP sintasa, como
        en la segunda corrida del paper).
    rich_medium : si es True, el caso de crecimiento permite captar todos los
        nutrientes, como en el paper; si no, usa el medio actual del modelo.
    weights : costo de eliminar cada dirección, con claves "RXN" (ambas
        direcciones) o ("RXN", "forward"/"backward"). Lo no listado cuesta 1.
        Ver `evidence_weights` para un criterio basado en el paper.
    big_m : cota para las variables duales alpha, beta.
    n_solutions : cuántas soluciones alternativas enumerar como máximo. Cada
        corte no-good prohíbe una solución y también sus superconjuntos, así
        que salen solo alternativas minimales, de menor a mayor tamaño; la
        lista termina antes si no quedan más.

    Cada candidato del MILP se verifica con LPs exactos (crecimiento >= T y
    sin EGCs) antes de devolverlo; los que fallan por tolerancias numéricas se
    descartan con un corte y se vuelve a resolver.

    Devuelve una lista de resultados, uno por solución encontrada (o uno solo
    con el estado del solver si el problema es infactible). `growth` es el
    crecimiento verificado.
    """
    energy_rxns = _as_list(energy_rxns)
    biomass_rxn = biomass_rxn or _biomass_id(model)
    rxns = list(model.reactions)
    n = len(rxns)
    idx = {r.id: j for j, r in enumerate(rxns)}
    S = csr_matrix(create_stoichiometric_matrix(model, array_type="lil"))
    m = S.shape[0]

    lb = np.array([r.lower_bound for r in rxns], dtype=float)
    ub = np.array([r.upper_bound for r in rxns], dtype=float)

    # Cotas del problema interno: sin captación de nutrientes y sin
    # mantenimiento forzado; se relaja todo a l <= 0 <= u.
    l_in = np.minimum(lb, 0)
    u_in = np.maximum(ub, 0)
    for r in model.boundary:
        l_in[idx[r.id]] = 0
    for rid in energy_rxns:
        l_in[idx[rid]] = 0

    if candidates is None:
        excluded = {r.id for r in model.boundary} | {biomass_rxn, *energy_rxns}
        candidates = [r.id for r in rxns if r.id not in excluded]
    cand = set(candidates) - set(protected)

    if rich_medium:
        for r in model.exchanges:
            lb[idx[r.id]] = min(lb[idx[r.id]], -1000)

    # Binarias: una por dirección que existe en la reacción.
    fwd = [j for j in range(n) if rxns[j].id in cand and ub[j] > 0]
    bwd = [j for j in range(n) if rxns[j].id in cand and lb[j] < 0]
    kf, kb = len(fwd), len(bwd)

    # Orden de variables: [v (n) | lambda (m) | alpha (n) | beta (n) | dF (kf) | dB (kb)]
    o_v, o_lam, o_a, o_b = 0, n, n + m, 2 * n + m
    o_df = 3 * n + m
    o_db = o_df + kf
    nvar = o_db + kb

    var_lb = np.zeros(nvar)
    var_ub = np.zeros(nvar)
    var_lb[o_v:o_v + n], var_ub[o_v:o_v + n] = lb, ub
    var_lb[o_v + idx[biomass_rxn]] = max(lb[idx[biomass_rxn]], min_growth)
    var_lb[o_lam:o_lam + m], var_ub[o_lam:o_lam + m] = -np.inf, np.inf
    var_ub[o_a:o_a + n] = big_m
    var_ub[o_b:o_b + n] = big_m
    var_ub[o_df:] = 1

    # Si una dirección está permitida en el problema interno y no se puede
    # eliminar, su dual queda fijo en 0 (holgura complementaria sin binaria).
    fwd_set, bwd_set = set(fwd), set(bwd)
    for j in range(n):
        if u_in[j] > 0 and j not in fwd_set:
            var_ub[o_a + j] = 0
        if l_in[j] < 0 and j not in bwd_set:
            var_ub[o_b + j] = 0

    integrality = np.zeros(nvar)
    integrality[o_df:] = 1

    def block(rows, cols, vals, shape):
        return csr_matrix((vals, (rows, cols)), shape=shape)

    def zeros(r, c):
        return csr_matrix((r, c))

    rows, rhs_lo, rhs_hi = [], [], []

    # (1) Balance de masa en crecimiento: S v = 0
    rows.append(hstack([S, zeros(m, nvar - n)]))
    rhs_lo.append(np.zeros(m)); rhs_hi.append(np.zeros(m))

    # (2) Acoplamiento de flujos de crecimiento con las eliminaciones
    #     v_j + ub_j dF_j <= ub_j      y      v_j + lb_j dB_j >= lb_j
    if kf:
        a = block(range(kf), fwd, np.ones(kf), (kf, n))
        d = block(range(kf), range(kf), ub[fwd], (kf, kf))
        rows.append(hstack([a, zeros(kf, o_df - n), d, zeros(kf, kb)]))
        rhs_lo.append(np.full(kf, -np.inf)); rhs_hi.append(ub[fwd])
    if kb:
        a = block(range(kb), bwd, np.ones(kb), (kb, n))
        d = block(range(kb), range(kb), lb[bwd], (kb, kb))
        rows.append(hstack([a, zeros(kb, o_db - n), d]))
        rhs_lo.append(lb[bwd]); rhs_hi.append(np.full(kb, np.inf))

    # (3) Factibilidad dual del problema interno: S^T lambda + alpha - beta = c
    c = np.zeros(n)
    for rid in energy_rxns:
        c[idx[rid]] = 1
    I = identity(n, format="csr")
    rows.append(hstack([zeros(n, n), S.T, I, -I, zeros(n, kf + kb)]))
    rhs_lo.append(c); rhs_hi.append(c)

    # (4) Holgura complementaria vía Big-M: alpha_j <= M dF_j, beta_j <= M dB_j
    if kf:
        a = block(range(kf), fwd, np.ones(kf), (kf, n))
        d = block(range(kf), range(kf), np.full(kf, -big_m), (kf, kf))
        rows.append(hstack([zeros(kf, o_a), a, zeros(kf, n), d, zeros(kf, kb)]))
        rhs_lo.append(np.full(kf, -np.inf)); rhs_hi.append(np.zeros(kf))
    if kb:
        a = block(range(kb), bwd, np.ones(kb), (kb, n))
        d = block(range(kb), range(kb), np.full(kb, -big_m), (kb, kb))
        rows.append(hstack([zeros(kb, o_b), a, zeros(kb, kf), d]))
        rhs_lo.append(np.full(kb, -np.inf)); rhs_hi.append(np.zeros(kb))

    weights = weights or {}

    def weight(j: int, direction: str) -> float:
        rid = rxns[j].id
        w = weights.get((rid, direction), weights.get(rid, 1.0))
        if w < 0:
            raise ValueError(f"Peso negativo para {rid} ({direction}): {w}")
        return w

    cost = np.zeros(nvar)
    cost[o_df:o_db] = [weight(j, "forward") for j in fwd]
    cost[o_db:] = [weight(j, "backward") for j in bwd]

    def decode(x) -> list[Removal]:
        out = [Removal(rxns[j].id, "forward") for t, j in enumerate(fwd) if x[o_df + t] > 0.5]
        out += [Removal(rxns[j].id, "backward") for t, j in enumerate(bwd) if x[o_db + t] > 0.5]
        return sorted(out, key=lambda r: (r.reaction, r.direction))

    def check(removals: list[Removal]) -> tuple[float, bool]:
        """Crecimiento y ausencia de EGCs con LPs exactos sobre el delta
        redondeado. El MILP no basta: con delta = 1 - 1e-6 (dentro de la
        tolerancia de integralidad) una reacción "eliminada" aún deja pasar
        ub * 1e-6 de flujo, y en modelos grandes eso puede fingir crecimiento."""
        with model:
            apply_removals(model, removals)
            egc_free = all(v == 0 for v in detect_egcs(model, energy_rxns).values())
            if rich_medium:
                for r in model.exchanges:
                    r.lower_bound = min(r.lower_bound, -1000)
            model.objective = biomass_rxn
            growth = model.slim_optimize(error_value=0.0)
        return growth, egc_free

    def add_cut(chosen: np.ndarray, exact: bool) -> None:
        # Sin `exact` prohíbe el conjunto y sus superconjuntos; con `exact`,
        # solo ese conjunto (un superconjunto podría ser válido).
        cut = np.zeros(nvar)
        cut[o_df:] = np.where(chosen, 1.0, -1.0 if exact else 0.0)
        rows.append(csr_matrix(cut))
        rhs_lo.append(np.array([-np.inf])); rhs_hi.append(np.array([chosen.sum() - 1]))

    results = []
    rejected = 0
    while len(results) < n_solutions:
        A = vstack(rows, format="csr")
        res = milp(
            cost,
            constraints=LinearConstraint(A, np.concatenate(rhs_lo), np.concatenate(rhs_hi)),
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub),
            options={"time_limit": time_limit, "disp": verbose},
        )
        if res.x is None:
            # Infactible desde el inicio: se informa. Si ya hubo soluciones,
            # solo significa que no quedan alternativas.
            if not results:
                results.append(GlobalFitResult(status=res.message))
            break
        chosen = np.round(res.x[o_df:]).astype(bool)
        removals = decode(res.x)
        growth, egc_free = check(removals)
        grows = growth >= min_growth - TOL

        if not (grows and egc_free):
            # Artefacto numérico del MILP: se descarta y se vuelve a resolver.
            rejected += 1
            if rejected > MAX_REJECTED:
                results.append(GlobalFitResult(
                    status=f"se descartaron {rejected} candidatos que no pasaron la verificación"))
                break
            add_cut(chosen, exact=grows)
            continue

        results.append(GlobalFitResult(
            status="optimal" if res.status == 0 else res.message,
            removals=removals,
            objective=round(res.fun, 6),
            growth=growth,
            mip_gap=getattr(res, "mip_gap", None),
        ))
        if res.status != 0 or not chosen.any():
            break
        add_cut(chosen, exact=False)

    return results


def verify(
    model: Model,
    removals: list[Removal],
    energy_rxns: str | list[str] = "ATPM",
    rich_medium: bool = False,
) -> tuple[dict[str, float], float]:
    """Comprueba una solución con FBA independiente.

    Devuelve (energía máxima sin nutrientes por reacción, crecimiento máximo)
    del modelo corregido. Una solución correcta tiene todas las energías en 0.
    """
    with model:
        apply_removals(model, removals)
        energy = detect_egcs(model, energy_rxns)
        if rich_medium:
            for rxn in model.exchanges:
                rxn.lower_bound = min(rxn.lower_bound, -1000)
        growth = model.slim_optimize(error_value=np.nan)
    return energy, growth
