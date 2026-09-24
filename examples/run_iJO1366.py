"""Reproduce el caso de Fritzemeier et al. (2017) sobre iJO1366.

El modelo publicado ya está curado: los autores bloquearon seis reacciones
que formaban EGCs. Aquí se reactivan para recrear el problema y se deja que
GlobalFit encuentre la corrección mínima.
"""

import time
from pathlib import Path

from globalfit import detect_egcs, globalfit, load_bigg_model, verify

EGC_REACTIONS = ["SPODM", "SPODMpp", "SUCASPtpp", "SUCFUMtpp", "SUCMALtpp", "SUCTARTtpp"]

model = load_bigg_model("iJO1366", Path(__file__).resolve().parent.parent / "modelos")
print(f"Modelo curado: ATP sin nutrientes = {detect_egcs(model)['ATPM']:g}")

for rid in EGC_REACTIONS:
    model.reactions.get_by_id(rid).bounds = (-1000, 1000)
print(f"Con las 6 reacciones reactivadas: ATP sin nutrientes = "
      f"{detect_egcs(model)['ATPM']:g}")

start = time.time()
results = globalfit(model, n_solutions=3, time_limit=900)
print(f"\nGlobalFit terminó en {time.time() - start:.1f} s\n")

for i, result in enumerate(results, 1):
    energy, growth = verify(model, result.removals)
    print(f"Solución {i} ({result.status}): {result.objective} cambios")
    for removal in result.removals:
        print(f"   - {removal}")
    print(f"   verificación: ATP sin nutrientes = {energy['ATPM']:g}, crecimiento = {growth:.3f}\n")
