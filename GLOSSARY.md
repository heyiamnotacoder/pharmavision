# PharmaVision

Scoring rodent behaviour tests from overhead video, from the recorded test to a reported study result.

## Language

**Run**:
One processing of one video through the pipeline. Processing the same video again is a new run.
_Avoid_: Job, analysis, session

**Trial**:
One animal tested once in one behaviour test. The trial is the unit that enters the statistics; in the open field MVP, one video holds exactly one trial.
_Avoid_: Run, video, subject, session

**Study**:
A set of trials together with their design: groups, treatment, dose and blinding.
_Avoid_: Experiment, batch, project

**Gap**:
A stretch of consecutive frames where the tracker found no animal. Short gaps are bridged by linear interpolation; longer ones stay missing in every metric and go to QC.
_Avoid_: Dropout, missing frame (for the whole run)

**Flag**:
A QC finding over a frame range, with a reason and a severity. A warning is listed in the report; a critical flag leads to an exclusion.
_Avoid_: Alert, error, issue

**Exclusion**:
Frames, or a whole trial, left out of the statistics, with the reason and whether a rule or the model decided it. Every exclusion is listed; nothing is dropped silently.
_Avoid_: Rejection, filter, drop

**Rear candidate**:
A window of frames where the animal may be rearing: its mask shrank and changed shape while it stayed in place. Found by rules in the measure stage, which also scores it (see rear score); the behaviours stage shows it to Muse Spark for a second opinion.
_Avoid_: Rear, rearing event (for an unjudged window)

**Rear score**:
How likely a rear candidate holds a rear, from 0 to 1: a logistic over the candidate's mask features (duration, area drop, elongation change, speed, mask area), fitted on the train split's rater labels. The **rear count** is the candidates scoring at least the scorer's threshold. Measured, so it belongs to the measure stage; Muse Spark's label is a second opinion beside it.
_Avoid_: Rear probability (it is not calibrated per video), model rear count

**Behaviour event**:
One judged rear candidate: its frames, the model's label (rearing or none), its confidence and rationale, and whether a rule or the model produced it. Every candidate gets one, so each verdict can be checked; one under the minimum confidence is a low-confidence event, reported as such.
_Avoid_: Detection, prediction

**Consensus rear**:
A rear that at least two of the three Sturman raters labelled as the same kind (supported or unsupported), with a time IoU of at least 0.5. It spans the time they share. The ground truth for rearing recall, because single raters disagree on where rears start and end.
_Avoid_: True rear, gold rear
