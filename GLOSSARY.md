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

**Rear candidate**:
A window of frames where the animal may be rearing: its mask shrank and changed shape while it stayed in place. Found by rules in the measure stage, then judged by the behaviours stage; a candidate is never counted as a rear itself.
_Avoid_: Rear, rearing event (for an unjudged window)
