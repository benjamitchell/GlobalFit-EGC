"""Compara con Fritzemeier et al. (2017), tabla S2, en los modelos de BiGG
que el paper reporta con EGCs, más iJO1366 con las 6 reacciones reactivadas.

Sigue los métodos del paper: cota inferior de ATPM en 0, las 15 reacciones
de disipación de la tabla S1, crecimiento en medio rico y una segunda corrida
que no permite eliminar la ATP sintasa ("synthase").

Ojo: los modelos se descargan hoy de BiGG y pueden diferir de las versiones
de julio de 2015 que usó el paper.
"""

import time
from pathlib import Path

from globalfit import (
    add_energy_dissipation_reactions,
    detect_egcs,
    globalfit,
    load_bigg_model,
    verify,
)

MIN_GROWTH = 0.01  # el paper no reporta su T_g

# Tabla S2: eliminaciones sugeridas por el paper (forward / backward).
PAPER = {
    "iND750": {"simple": "ACOAH (forward)", "synthase": "ACOAH (forward)"},
    "iJN746": {"simple": "ACALD (forward)", "synthase": "ALDD2x_copy2 (backward)"},
    "iJO1366+6": {"simple": "MOX (backward) o SPODM (forward), según el texto", "synthase": "-"},
    "iRC1080": {"simple": "49 cambios (ver tabla S2)", "synthase": "49 cambios (ver tabla S2)"},
}
ATP_SYNTHASES = ["ATPS4rpp", "ATPS4r", "ATPS3m", "ATPS3v", "ATPS", "ATPSh"]
IJO_EGC = ["SPODM", "SPODMpp", "SUCASPtpp", "SUCFUMtpp", "SUCMALtpp", "SUCTARTtpp"]


MODELS_DIR = Path(__file__).resolve().parent.parent / "modelos"


def load(name):
    if name == "iJO1366+6":
        model = load_bigg_model("iJO1366", MODELS_DIR)
        for rid in IJO_EGC:
            model.reactions.get_by_id(rid).bounds = (-1000, 1000)
        return model
    model = load_bigg_model(name, MODELS_DIR)
    if name == "iRC1080":
        # El SBML no trae objetivo; el paper no dice cuál usó (las tres
        # biomasas dan correcciones de 49 cambios).
        model.objective = "BIOMASS_Chlamy_hetero"
    return model


for name, expected in PAPER.items():
    print(f"\n=== {name} ===")
    model = load(name)
    if "ATPM" in model.reactions:
        model.reactions.ATPM.lower_bound = 0
    edrs = add_energy_dissipation_reactions(model)

    found = {k.removeprefix("EDR_"): round(v, 2)
             for k, v in detect_egcs(model, edrs, normalize=True).items() if v > 0}
    print(f"EGCs detectados (normalizado): {found or 'ninguno'}")
    if not found:
        continue

    synthases = [r for r in ATP_SYNTHASES if r in model.reactions]
    for run, protected in [("simple", []), ("synthase", synthases)]:
        start = time.time()
        results = globalfit(model, energy_rxns=edrs, rich_medium=True,
                            min_growth=MIN_GROWTH, protected=protected,
                            n_solutions=1 if name == "iRC1080" else 5)
        elapsed = time.time() - start
        print(f"\n  [{run}] {elapsed:.1f} s — paper: {expected[run]}")
        for result in results:
            if result.status != "optimal":
                print(f"    {result.status}")
                continue
            energy, growth = verify(model, result.removals, edrs, rich_medium=True)
            ok = "OK" if all(v == 0 for v in energy.values()) else f"FALLA {energy}"
            removals = ", ".join(map(str, result.removals))
            print(f"    {len(result.removals)} cambio(s): {removals}  | verif {ok}, crecimiento {growth:.3f}")
