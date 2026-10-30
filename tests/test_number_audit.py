"""The number audit through its two seams: `known_values` flattens what the report agent's tools
returned into numbers with their source, and `audit_numbers` checks every number in the prose
against them. Expected numbers are literals worked out by hand or by scipy directly, never read
back from the tables under test."""

from pathlib import Path

from pharmavision.artifacts import AuditSettings, KnownValue, NumberAudit
from pharmavision.audit import audit_numbers, known_values
from pharmavision.pipeline import study_stats

from .studies import write_study

VEHICLE = [10.1, 11.3, 9.8, 10.6, 12.0, 10.9]
DIAZEPAM = [12.4, 13.1, 11.8, 12.9, 14.0, 12.2]

# scipy on the fixture directly: means 10.783 and 12.733 cm, SDs 0.804 and 0.779, Student's
# t(10) = 4.268, p = 0.00164, difference 1.95 cm, 95% CI 0.932 to 2.968.
HONEST = """\
## Results

This section follows ARRIVE 2.0 (Essential 10). Mice received saline or diazepam (1 mg/kg).
Distance travelled was 10.78 ± 0.80 cm in the vehicle group (n = 6) and 12.73 ± 0.78 cm in
the diazepam group (n = 6; trials B07 to B12). Diazepam increased distance travelled by 1.95 cm
(95% CI 0.93 to 2.97; Student's t(10) = 4.27, p = 1.6 × 10⁻³).
"""


def known(**values: float) -> list[KnownValue]:
    return [KnownValue(source=source, value=value) for source, value in values.items()]


def audit(prose: str, **values: float) -> NumberAudit:
    return audit_numbers(prose, known(**values), AuditSettings())


def test_honest_prose_about_a_study_passes(tmp_path: Path, runs_dir: Path) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": VEHICLE, "diazepam": DIAZEPAM})
    stats = study_stats(study, runs_dir)
    tables = {"design": study.design(), "trials": study.trials(), "stats": stats}

    result = audit_numbers(HONEST, known_values(tables), AuditSettings())

    assert result.passed, [n.text for n in result.unmatched]
    assert [n.text for n in result.numbers] == [
        "1", "10.78", "0.80", "6", "12.73", "0.78", "6", "1.95", "95%", "0.93", "2.97", "10",
        "4.27", "1.6 × 10⁻³",
    ]  # fmt: skip
    by_text = {n.text: n.source for n in result.numbers}
    assert by_text["4.27"] == "stats.metrics[0].comparison.statistic"
    assert by_text["95%"] == "stats.settings.ci_level"
    assert by_text["6"] == "stats.metrics[0].groups[0].n"  # a field, not a list's length


def test_an_invented_number_fails_and_is_named_with_its_sentence() -> None:
    prose = "Diazepam increased distance travelled. This is a 38% reduction in immobility."

    result = audit(prose, distance=12.73, immobility=41.2)

    assert not result.passed
    [invented] = result.unmatched
    assert invented.text == "38%"
    assert invented.sentence == "This is a 38% reduction in immobility."
    start, end = invented.sentence_span
    assert prose[start:end] == invented.sentence


def test_a_value_rounded_to_the_written_precision_passes() -> None:
    passed = audit("Mice ran 3413 cm, or 3,412.6 cm to one decimal.", distance=3412.6)
    too_far = audit("Mice ran 3414 cm.", distance=3412.6)

    assert passed.passed
    assert [n.text for n in too_far.unmatched] == ["3414"]


def test_scientific_notation_is_rounded_at_its_mantissa() -> None:
    passed = audit("p = 1.6e-3, or 1.6 x 10^-3, or 1.6 × 10⁻³.", p=0.001641)
    too_far = audit("p = 1.7 × 10⁻³.", p=0.001641)

    assert passed.passed
    assert [n.text for n in too_far.unmatched] == ["1.7 × 10⁻³"]


def test_a_percent_matches_a_fraction_times_100_not_a_value_in_other_units() -> None:
    result = audit(
        "Mice spent 72.4% of the time near the wall and 38% in the centre.",
        thigmotaxis=0.724,
        centre_time_s=38.2,
    )

    assert [n.text for n in result.unmatched] == ["38%"]


def test_signs_are_not_audited_so_a_difference_reads_either_way() -> None:
    result = audit("Distance fell by 1.95 cm (difference −1.95 cm).", difference=-1.95)

    assert result.passed


def test_identifiers_and_allowed_phrases_are_not_audited() -> None:
    result = audit("Trials B07, OFT_16 and OFT-16 (cm^2) were scored by SAM 3.1 per ARRIVE 2.0.")

    assert result.numbers == []


def test_with_nothing_known_every_number_fails() -> None:
    result = audit("Six mice per group, n = 6.")

    assert [n.text for n in result.unmatched] == ["6"]


def test_qc_counts_are_known_but_numbers_inside_model_written_reasons_are_not() -> None:
    window = {"scope": "window", "first_frame": 30, "last_frame": 44, "source": "model"}
    qc = {
        "exclusions": [
            {**window, "reason": "a hand was in the arena for 4.2 s"},
            {**window, "reason": "a glove reached in"},
        ]
    }

    prose = "Two windows (2) were excluded: a hand for 4.2 s."

    result = audit_numbers(prose, known_values({"qc": qc}), AuditSettings())

    by_text = {n.text: n.source for n in result.numbers}
    assert by_text == {"2": "len(qc.exclusions)", "4.2": None}


def test_the_audit_is_a_typed_artifact_that_round_trips() -> None:
    result = audit("A 38% reduction.", distance=12.73)

    again = NumberAudit.model_validate_json(result.model_dump_json())

    assert again == result
    assert not again.passed


def test_each_markdown_table_row_and_heading_is_its_own_sentence() -> None:
    prose = "## Results 2\n| vehicle | 38 |\nMice were tested.\n- 3 trials were excluded"

    result = audit(prose)

    assert [n.sentence for n in result.unmatched] == [
        "## Results 2", "| vehicle | 38 |", "- 3 trials were excluded",
    ]  # fmt: skip


def test_a_sentence_runs_on_past_abbreviations_and_wrapped_lines() -> None:
    prose = "As in Sturman et al. the mice ran 99 cm, a difference of\n-1.95 cm and\n*p* = 0.04."

    result = audit(prose)

    assert {n.sentence for n in result.unmatched} == {prose}
