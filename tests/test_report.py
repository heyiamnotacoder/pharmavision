"""The report agent through the study-level seam: Muse Spark reads the study through read-only
tools and writes an ARRIVE 2.0 Essential 10 methods and results section, which ships only after
the number audit. Model responses are replayed, never called: a stream recorded through the
production judge is replayed by the recorded judge by the same cache key."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from pharmavision.artifacts import BehaviourEvent, Events
from pharmavision.audit import known_values
from pharmavision.judge.muse_spark import (
    JudgeResponseError,
    MuseSparkJudge,
    MuseSparkSettings,
    RecordedJudge,
)
from pharmavision.pipeline import (
    RunFolder,
    StudyFolder,
    StudyNotReadyError,
    study_report,
    study_stats,
)

from .judges import ScriptedJudge
from .muse_replay import function_outputs, text, tool_calls
from .sam_replay import ReplayTransport
from .studies import write_study

VEHICLE = [10.1, 11.3, 9.8, 10.6, 12.0, 10.9]
DIAZEPAM = [12.4, 13.1, 11.8, 12.9, 14.0, 12.2]
TOOLS = ["get_protocol", "get_metrics_table", "get_stats", "get_qc", "get_events_summary"]

# scipy on the fixture directly: means 10.783 and 12.733 cm, SDs 0.804 and 0.779, Student's
# t(10) = 4.268, p = 0.00164, difference 1.95 cm, 95% CI 0.932 to 2.968.
RESULTS = """\
Distance travelled was 10.78 ± 0.80 cm in the vehicle group (n = 6) and 12.73 ± 0.78 cm in
the diazepam group (n = 6). Diazepam increased distance travelled by 1.95 cm (95% CI 0.93 to
2.97; Student's t(10) = 4.27, p = 0.0016). The pipeline makes no rear count."""

HONEST = f"""\
## Study design

Two groups were compared: saline and diazepam at 1 mg/kg. Each trial was one mouse.

## Sample size

Each group had 6 mice, 12 in total.

## Inclusion and exclusion criteria

No trial was excluded.

## Blinding

Trials were scored automatically under blinding codes B01 to B12.

## Outcome measures

Distance travelled was measured from the tracked centroid in a 45 cm square arena.

## Statistical methods

A fixed rule chose the test from Shapiro-Wilk tests at 0.05 and Levene's test.

## Experimental animals

Mice were filmed from above.

## Experimental procedures

Each mouse was filmed for 3 s.

## Results

{RESULTS}
"""


def fixture_study(tmp_path: Path, runs_dir: Path) -> StudyFolder:
    study = write_study(tmp_path, runs_dir, {"vehicle": VEHICLE, "diazepam": DIAZEPAM})
    study_stats(study, runs_dir)
    return study


def test_a_report_through_the_recorded_judge_passes_the_audit(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = fixture_study(tmp_path, runs_dir)
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    transport = ReplayTransport([tool_calls("get_protocol", "get_stats"), text(HONEST)])
    study_report(study, runs_dir, MuseSparkJudge(settings, transport))

    report = study_report(study, runs_dir, RecordedJudge(settings))

    assert report.passed
    assert len(report.drafts) == 1
    assert report.model_id == "muse-spark-1.3"
    assert "12.73 ± 0.78 cm" in report.markdown
    assert study.report() == report
    assert (study.path / "report.md").read_text() == report.markdown
    first, second = transport.requests
    assert [t["name"] for t in first["tools"]] == TOOLS
    assert {json.dumps(t["parameters"]) for t in first["tools"]} == {
        '{"type": "object", "properties": {}, "required": [], "additionalProperties": false}'
    }
    outputs = function_outputs(second)
    assert sorted(outputs) == ["call-0", "call-1"]
    assert json.loads(outputs["call-1"])["metrics"][0]["comparison"]["test"] == "student_t"


INVENTED = "Diazepam raised distance travelled by 18% over vehicle."


def test_an_invented_number_re_prompts_the_agent_with_its_sentence(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = fixture_study(tmp_path, runs_dir)
    first = HONEST.replace(RESULTS, f"{RESULTS} {INVENTED}")
    transport = ReplayTransport([tool_calls("get_stats"), text(first), text(HONEST)])
    judge = MuseSparkJudge(MuseSparkSettings(cache_dir=tmp_path / "cache"), transport)

    report = study_report(study, runs_dir, judge)

    assert report.passed
    assert [d.audit.passed for d in report.drafts] == [False, True]
    assert [n.text for n in report.drafts[0].audit.unmatched] == ["18%"]
    assert report.struck == []
    assert INVENTED not in report.markdown
    re_prompt = transport.requests[2]["input"]
    assert re_prompt[1]["role"] == "assistant"
    assert re_prompt[1]["content"][0]["text"] == first
    rejection = re_prompt[2]["content"][0]["text"]
    assert "'18%'" in rejection
    assert INVENTED in rejection


def test_a_second_failure_ships_the_report_with_its_sentence_struck_out(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = fixture_study(tmp_path, runs_dir)
    invented = HONEST.replace(RESULTS, f"{RESULTS} {INVENTED}")
    transport = ReplayTransport([text(invented), text(invented)])
    judge = MuseSparkJudge(MuseSparkSettings(cache_dir=tmp_path / "cache"), transport)

    report = study_report(study, runs_dir, judge)

    assert not report.passed
    assert len(report.drafts) == 2
    assert [n.text for n in report.struck] == ["18%"]
    assert f"~~{INVENTED}~~ **[failed the number audit]**" in report.markdown
    assert "12.73 ± 0.78 cm" in report.markdown
    assert "~~" not in report.markdown.replace(f"~~{INVENTED}~~", "")
    assert study.report() == report


def section(markdown: str, heading: str) -> str:
    """The text under `## heading` in `markdown`, up to the next heading."""
    return markdown.split(f"## {heading}\n\n", 1)[1].split("\n## ", 1)[0].strip()


def test_housing_and_randomisation_are_left_to_fill_whatever_the_agent_writes(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = fixture_study(tmp_path, runs_dir)
    guessed = HONEST.replace(
        "## Blinding", "## Randomisation\n\nMice were randomised by coin toss.\n\n## Blinding"
    ).replace("## Results\n\n" + RESULTS, "")
    judge = ScriptedJudge(drafts=[guessed])

    report = study_report(study, runs_dir, judge)

    assert section(report.markdown, "Randomisation") == (
        "[TO FILL: how animals were allocated to groups]\n\n"
        "[TO FILL: the order in which animals were tested]"
    )
    assert "coin toss" not in report.markdown
    animals = section(report.markdown, "Experimental animals")
    assert animals.endswith("[TO FILL: housing and husbandry]")
    assert animals.startswith("Mice were filmed from above.")
    assert section(report.markdown, "Results") == "[TO FILL: nothing was written for this item]"
    assert report.passed


def folder_bytes(path: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_the_tools_read_the_stored_tables_and_compute_no_number(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(
        tmp_path, runs_dir, {"vehicle": VEHICLE, "diazepam": DIAZEPAM},
        excluded=frozenset({"diazepam-0"}),
    )  # fmt: skip
    study_stats(study, runs_dir)
    unsure = BehaviourEvent(
        first_frame=25, last_frame=39, label="rearing", confidence=0.42, low_confidence=True,
        rationale="Unclear.", source="model",
    )  # fmt: skip
    RunFolder(runs_dir / "vehicle-0").write_events(Events(events=[unsure]))
    judge = ScriptedJudge(drafts=[HONEST, HONEST])
    study_report(study, runs_dir, judge)
    before = folder_bytes(tmp_path)

    brief = judge.briefs[0]
    outputs = {tool.name: json.loads(tool.read()) for tool in brief.tools}

    assert list(outputs) == TOOLS
    assert folder_bytes(tmp_path) == before
    runs = [RunFolder(runs_dir / t.run_id) for t in study.design().trials]
    stored: dict[str, BaseModel] = {"design": study.design(), "trials": study.trials()}
    stored["stats"] = study.stats()
    for n, run in enumerate(runs):
        stored |= {f"manifest{n}": run.manifest(), f"metrics{n}": run.metrics()}
        stored |= {f"qc{n}": run.qc(), f"events{n}": run.events()}
    stored_values = {k.value for k in known_values(stored)}
    told_values = {k.value for k in known_values(outputs)}
    in_rule_reason = {0.17, 0.05}  # "the animal was not found on 17% of frames (limit 5%)"
    assert told_values <= stored_values | in_rule_reason
    assert 0.42 not in told_values  # the model's own confidence is not a measured number
    [event] = outputs["get_events_summary"]["trials"][0]["low_confidence_events"]
    assert event == {
        "first_frame": 25, "last_frame": 39, "label": "rearing", "model_rationale": "Unclear.",
    }  # fmt: skip
    [excluded] = outputs["get_metrics_table"]["trials"][6]["exclusions"]
    assert excluded["reason"] == "the animal was not found on 17% of frames (limit 5%)"


def test_a_study_without_stats_gets_no_report(tmp_path: Path, runs_dir: Path) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": VEHICLE, "diazepam": DIAZEPAM})

    with pytest.raises(StudyNotReadyError, match="run the study stats first"):
        study_report(study, runs_dir, ScriptedJudge(drafts=[HONEST]))

    assert not (study.path / "report.json").exists()


def test_an_agent_that_keeps_calling_tools_writes_no_report(tmp_path: Path, runs_dir: Path) -> None:
    study = fixture_study(tmp_path, runs_dir)
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache", report_max_turns=2)
    transport = ReplayTransport([tool_calls("get_stats"), tool_calls("get_shell")])

    with pytest.raises(JudgeResponseError, match="no report after 2 requests"):
        study_report(study, runs_dir, MuseSparkJudge(settings, transport))

    assert not (study.path / "report.json").exists()
    assert transport.streams == []


def test_a_rule_exclusion_can_be_reported_with_the_numbers_in_its_reason(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(
        tmp_path, runs_dir, {"vehicle": VEHICLE, "diazepam": DIAZEPAM},
        excluded=frozenset({"diazepam-0"}),
    )  # fmt: skip
    study_stats(study, runs_dir)
    quoted = "Trial B07 was excluded: the animal was not found on 17% of frames (limit 5%)."
    draft = HONEST.replace("No trial was excluded.", quoted).replace(RESULTS, "")

    report = study_report(study, runs_dir, ScriptedJudge(drafts=[draft]))

    assert report.passed, [n.text for n in report.struck]
    assert quoted in report.markdown


def test_a_study_whose_runs_were_filmed_differently_gets_no_report(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = fixture_study(tmp_path, runs_dir)
    odd = RunFolder(runs_dir / "diazepam-2")
    manifest = odd.manifest()
    video = manifest.video.model_copy(update={"fps": 25.0})
    odd.write_manifest(manifest.model_copy(update={"video": video}))

    with pytest.raises(StudyNotReadyError, match="B09"):
        study_report(study, runs_dir, ScriptedJudge(drafts=[HONEST]))

    assert not (study.path / "report.json").exists()
