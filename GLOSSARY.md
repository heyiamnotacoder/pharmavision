# PharmaVision

Scoring rodent behaviour tests from overhead video, from the recorded test to a reported study result.

## Language

**Run**:
One processing of one video through the pipeline. Processing the same video again is a new run.
_Avoid_: Job, analysis, session

**Job**:
The API working on a run or a study in the background: a run's stages up to the arena gate, then the stages after it, or a study's report. Its status (queued, running, awaiting zone confirmation, done, failed) is kept in SQLite. A run is what is processed; a job is the processing in progress, so a run is never called a job.
_Avoid_: Task, worker

**Arena gate**:
The one human check in the pipeline. A run stops after the arena stage until a person confirms its zones in the zone editor, where they may move the floor's corners or draw them. Only the floor is ever edited: the centre zone and the scale (px/cm) are derived from it.
_Avoid_: Approval, review (the model reviews QC flags, not zones)

**Empty arena**:
The arena a run waits at the gate with when SAM 3.1 found no usable floor: no floor, no centre zone, no scale. It cannot be confirmed until a floor is drawn.
_Avoid_: Failed arena, missing arena

**Trial**:
One animal tested once in one behaviour test. The trial is the unit that enters the statistics; in the open field MVP, one video holds exactly one trial.
_Avoid_: Run, video, subject, session

**Study**:
A set of trials together with their design: groups, treatment, dose and blinding.
_Avoid_: Experiment, batch, project

**Reference group**:
The first group in a study's design, usually the vehicle. Two-group differences are the other group minus it, and post-hoc pairs run in the design's order from it.
_Avoid_: Control (when it is not a control), baseline

**Abstain**:
To give no number where the data cannot support one, and say why: the stats abstain on a metric when a group has too few trials left or no trial differs; the report writes `[TO FILL]` for a fact the system cannot know.
_Avoid_: Fail, skip, zero

**Gap**:
A stretch of consecutive frames where the tracker found no animal. Short gaps are bridged by linear interpolation; longer ones stay missing in every metric and go to QC.
_Avoid_: Dropout, missing frame (for the whole run)

**Flag**:
A QC finding over a frame range, with a reason and a severity. A warning is listed in the report; a critical flag leads to an exclusion.
_Avoid_: Alert, error, issue

**Exclusion**:
Frames, or a whole trial, left out of the statistics, with the reason and whether a rule or the model decided it. Every exclusion is listed; nothing is dropped silently.
_Avoid_: Rejection, filter, drop

**Review**:
The models' look at one flagged window (a warning): SAM 3.1's "hand" and "glove" prompts on a few of its frames, then Muse Spark's verdict on a short clip around it: keep it, exclude the window, or exclude the trial, with a reason. Verdicts apply automatically and every one is listed, kept or not. Model review can be turned off, leaving the deterministic flags alone.
_Avoid_: Approval, override (a human's, stretch goal FR-S6)

**Rear candidate**:
A window of frames where the animal may be rearing: its mask shrank and changed shape while it stayed in place. Found by rules in the measure stage, which also scores it (see rear score); the behaviours stage shows it to Muse Spark for a second opinion.
_Avoid_: Rear, rearing event (for an unjudged window)

**Rear score**:
How likely a rear candidate holds a rear, from 0 to 1: a logistic over the candidate's mask features (duration, area drop, elongation change, speed, mask area), fitted on the train split's rater labels. Measured, so it belongs to the measure stage; Muse Spark's label is a second opinion beside it. Evidence per window only: it is never summed or thresholded into a rear count (ADR 0003).
_Avoid_: Rear probability (it is not calibrated per video), rear count

**Behaviour event**:
One judged rear candidate: its frames, the model's label (rearing or none), its confidence and rationale, and whether a rule or the model produced it. Every candidate gets one, so each verdict can be checked; one under the minimum confidence is a low-confidence event, reported as such.
_Avoid_: Detection, prediction

**Consensus rear**:
A rear that at least two of the three Sturman raters labelled as the same kind (supported or unsupported), with a time IoU of at least 0.5. It spans the time they share. The ground truth for rearing recall, because single raters disagree on where rears start and end.
_Avoid_: True rear, gold rear

**Number audit**:
The blocking check on report prose: every number in it must round to a value in the computed tables (metrics, stats, QC counts), or the prose fails and the unmatched numbers are listed with their sentences. Numbers inside model-written text are never taken as known.
_Avoid_: Fact check, validation (for this gate)
