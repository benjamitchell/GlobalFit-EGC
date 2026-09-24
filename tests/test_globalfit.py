import pytest
from cobra import Reaction
from cobra.io import load_model

from globalfit import (
    BIGG_DISSIPATION_REACTIONS,
    MODELSEED_DISSIPATION_REACTIONS,
    add_dissipation_reaction,
    add_energy_dissipation_reactions,
    detect_egcs,
    evidence_weights,
    globalfit,
    verify,
)

# Pares de reacciones irreversibles del modelo core que, al hacerse
# reversibles, crean un EGC (encontrados barriendo todos los pares).
EGC_PAIRS = [("SUCCt2_2", "SUCCt3"), ("FBP", "PFK"), ("GLNS", "GLUN")]

NADH_DISSIPATION = BIGG_DISSIPATION_REACTIONS["NADH"]


@pytest.fixture
def textbook():
    return load_model("textbook")


def inject(model, pairs):
    for pair in pairs:
        for rid in pair:
            model.reactions.get_by_id(rid).lower_bound = -1000


def add_nadh_egc(model):
    """Reducción oaa + 2 H+ -> mal sin donante de electrones: con MDH genera
    NADH de la nada, pero en el modelo core no se traduce en ATP. Está
    balanceada en masa y le faltan 2 electrones, igual que la EDR de NADH."""
    fake = Reaction("FAKE_OAA", lower_bound=0, upper_bound=1000)
    model.add_reactions([fake])
    m = model.metabolites
    fake.add_metabolites({m.oaa_c: -1, m.h_c: -2, m.mal__L_c: 1})


def assert_clean(energy):
    assert all(v == 0 for v in energy.values()), energy


def test_clean_model_needs_no_changes(textbook):
    assert_clean(detect_egcs(textbook))
    [result] = globalfit(textbook)
    assert result.status == "optimal"
    assert result.removals == []


@pytest.mark.parametrize("pair", EGC_PAIRS)
def test_single_egc_is_removed_with_one_change(textbook, pair):
    inject(textbook, [pair])
    assert detect_egcs(textbook)["ATPM"] > 0

    [result] = globalfit(textbook)
    assert result.objective == 1
    energy, growth = verify(textbook, result.removals)
    assert_clean(energy)
    assert growth >= 0.1


def test_three_egcs_and_alternative_solutions(textbook):
    inject(textbook, EGC_PAIRS)
    results = globalfit(textbook, n_solutions=4)

    assert len(results) == 4
    seen = set()
    for result in results:
        assert result.objective == 3
        energy, growth = verify(textbook, result.removals)
        assert_clean(energy)
        assert growth >= 0.1
        seen.add(tuple(map(str, result.removals)))
    assert len(seen) == 4  # los cortes no-good dan soluciones distintas


def test_growth_requirement_is_respected(textbook):
    inject(textbook, EGC_PAIRS)
    [result] = globalfit(textbook, min_growth=0.85)
    _, growth = verify(textbook, result.removals)
    assert growth >= 0.85 - 1e-6


def test_nadh_egc_is_invisible_to_atp_only(textbook):
    add_nadh_egc(textbook)
    add_dissipation_reaction(textbook, "DISS_nadh", NADH_DISSIPATION)

    energy = detect_egcs(textbook, ["ATPM", "DISS_nadh"])
    assert energy["ATPM"] == 0
    assert energy["DISS_nadh"] > 0

    # Mirando solo ATP, GlobalFit no ve nada que corregir.
    [atp_only] = globalfit(textbook, energy_rxns="ATPM")
    assert atp_only.removals == []


def test_multiple_energy_metabolites_are_fixed_together(textbook):
    add_nadh_egc(textbook)
    inject(textbook, [EGC_PAIRS[0]])
    add_dissipation_reaction(textbook, "DISS_nadh", NADH_DISSIPATION)
    energy_rxns = ["ATPM", "DISS_nadh"]

    before = detect_egcs(textbook, energy_rxns)
    assert before["ATPM"] > 0 and before["DISS_nadh"] > 0

    [result] = globalfit(textbook, energy_rxns=energy_rxns)
    assert result.objective == 2
    energy, growth = verify(textbook, result.removals, energy_rxns)
    assert_clean(energy)
    assert growth >= 0.1


def test_dissipation_with_unknown_metabolite_fails_clearly(textbook):
    with pytest.raises(KeyError, match="gtp_c"):
        add_dissipation_reaction(textbook, "DISS_gtp", {"gtp_c": -1, "gdp_c": 1})


def test_paper_dissipation_reactions_skip_missing_metabolites(textbook):
    added = add_energy_dissipation_reactions(textbook)
    # El modelo core no tiene CTP, GTP, UTP, ITP, flavinas, menaquinonas ni
    # periplasma, así que solo entran estas seis.
    assert added == ["EDR_ATP", "EDR_NADH", "EDR_NADPH", "EDR_Q8H2", "EDR_ACCOA", "EDR_GLU"]
    assert all(v == 0 for v in detect_egcs(textbook, added).values())


def test_paper_dissipation_reactions_detect_and_fix(textbook):
    add_nadh_egc(textbook)
    inject(textbook, [EGC_PAIRS[0]])
    edrs = add_energy_dissipation_reactions(textbook)
    before = detect_egcs(textbook, edrs)
    assert before["EDR_ATP"] > 0 and before["EDR_NADH"] > 0

    [result] = globalfit(textbook, energy_rxns=edrs)
    energy, _ = verify(textbook, result.removals, edrs)
    assert_clean(energy)


def test_normalized_detection_adds_disjoint_cycles(textbook):
    single = []
    for pair in EGC_PAIRS:
        with textbook:
            inject(textbook, [pair])
            single.append(detect_egcs(textbook, normalize=True)["ATPM"])
    # El ciclo de succinato carga 0,75 ATP por vuelta; los otros, 1.
    assert single == pytest.approx([0.75, 1, 1])

    inject(textbook, EGC_PAIRS)
    assert detect_egcs(textbook, normalize=True)["ATPM"] == pytest.approx(sum(single))


def test_protected_reactions_are_never_removed(textbook):
    inject(textbook, EGC_PAIRS)
    results = globalfit(textbook, n_solutions=6, protected=["ATPS4r", "SUCCt3"])
    # Solo existen 4 alternativas minimales; la enumeración se detiene sola.
    assert len(results) == 4
    for result in results:
        assert result.status == "optimal"
        assert not {r.reaction for r in result.removals} & {"ATPS4r", "SUCCt3"}
        energy, _ = verify(textbook, result.removals)
        assert_clean(energy)


def test_rich_medium_growth_case(textbook):
    inject(textbook, EGC_PAIRS)
    [result] = globalfit(textbook, rich_medium=True, min_growth=1.0)
    _, growth = verify(textbook, result.removals, rich_medium=True)
    assert growth >= 1.0 - 1e-6


def test_weights_break_ties(textbook):
    inject(textbook, [EGC_PAIRS[0]])
    # Sin pesos hay empate entre las dos direcciones de transporte.
    tied = {str(r.removals[0]) for r in globalfit(textbook, n_solutions=2)}
    assert tied == {"SUCCt2_2 (backward)", "SUCCt3 (backward)"}

    [result] = globalfit(textbook, weights={"SUCCt3": 5})
    assert [str(r) for r in result.removals] == ["SUCCt2_2 (backward)"]
    assert result.objective == 1

    [result] = globalfit(textbook, weights={("SUCCt3", "backward"): 0.2})
    assert [str(r) for r in result.removals] == ["SUCCt3 (backward)"]
    assert result.objective == pytest.approx(0.2)


def test_weighted_solutions_are_enumerated_by_cost(textbook):
    inject(textbook, EGC_PAIRS)
    results = globalfit(textbook, weights=evidence_weights(textbook), n_solutions=5)
    costs = [r.objective for r in results]
    assert costs == sorted(costs)
    for result in results:
        energy, _ = verify(textbook, result.removals)
        assert_clean(energy)


def test_evidence_weights(textbook):
    w = evidence_weights(textbook, no_gene=0.5, irreversible=2.0)
    assert w[("PGI", "forward")] == 1.0  # reversible, con gen
    assert w[("PFK", "forward")] == 2.0  # irreversible, con gen
    no_gpr = next(r for r in textbook.reactions
                  if not r.gene_reaction_rule and r.reversibility and not r.boundary)
    assert w[(no_gpr.id, "backward")] == 0.5


def test_negative_weights_are_rejected(textbook):
    with pytest.raises(ValueError, match="negativo"):
        globalfit(textbook, weights={"PGI": -1})


def test_modelseed_dissipation_reactions_mirror_bigg():
    # Mismas 15 EDR, con los mismos coeficientes; solo cambian los ids.
    assert MODELSEED_DISSIPATION_REACTIONS.keys() == BIGG_DISSIPATION_REACTIONS.keys()
    for name, bigg in BIGG_DISSIPATION_REACTIONS.items():
        seed = MODELSEED_DISSIPATION_REACTIONS[name]
        assert sorted(seed.values()) == sorted(bigg.values()), name
        assert all(m.endswith(("_c0", "_p0")) for m in seed), name
