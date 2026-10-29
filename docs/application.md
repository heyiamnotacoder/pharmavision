# Meta Global AI Developer Hackathon — application draft

Deadline: **Oct 25, 11:59 PM ET** (closes earlier if 20,000 spots fill). Apply at
https://cerebralvalley.ai/e/meta-ai-developer-hackathon/apply

> The form is behind a login, so the exact questions weren't visible. These answers cover what
> the organisers say they score: **(1) a clear, feasible idea that fits a theme** and **(2) the
> technical skill shown in a past AI project**. Paste each block into the closest matching field and
> trim it if there's a character limit. Short versions are included.
>
> Fill in or check everything in `[brackets]` before you submit.

---

## Track
**Indie developer** (solo)

## Theme
**Push the Frontier**: making a scientific experiment faster and more reproducible.

---

## Project name
**PharmaVision**: an open-source AI lab for behavioural pharmacology

## Project idea: short version (~60 words)
PharmaVision turns a phone or webcam video of a rodent behaviour test (open field, elevated plus
maze, forced swim) into a scored, statistically analysed, ARRIVE-compliant results section. SAM 3.1
tracks the animal and segments the arena zones from a text prompt. Deterministic Python computes
every metric. Muse Spark classifies behaviours, flags bad trials and writes the report. It is an
open, auditable alternative to expensive commercial tracking software and to slow manual scoring.

## Project idea: full version
**Problem.** Preclinical drug studies depend on rodent behaviour tests: the open field (locomotion
and anxiety), the elevated plus maze (anxiolytics) and the forced swim test (antidepressants). Most
labs, including most pharmacology departments in India, score these by hand with a stopwatch or buy
expensive commercial software. Manual scoring is slow and differs between raters, which feeds
directly into the reproducibility crisis in behavioural neuroscience. Open-source pose tools such as
DeepLabCut need labelled training frames and ML know-how that a typical pharmacology lab doesn't
have. [Add one line from your own experience, e.g. "In our department we score these by hand in
practicals and dissertation work."]

**What I'll build.** A pipeline that takes video in and gives a results section out, built around a
split of work: **SAM 3.1 sees, Python measures, Muse Spark judges.**

1. **See (SAM 3.1).** A text prompt ("mouse") segments and tracks the animal through the whole video
   with no training data. The same model segments the arena from the first frame ("open arm",
   "closed arm", "centre zone"), and the known arena size gives a pixel-to-cm scale. SAM 3.1's
   multi-object tracking lets the same pipeline handle social interaction tests with two animals.
2. **Measure (deterministic Python, unit-tested).** From the masks it computes distance travelled,
   velocity, time and entries per zone, thigmotaxis, immobility in the forced swim test (frame-to-frame
   change in mask area below a threshold) and freezing. The LLM never computes a number.
3. **Judge (Muse Spark).**
   - Video understanding on short clips classifies rearing, grooming and head dips. These are
     ethological behaviours a centroid can't capture.
   - Quality control flags trials where tracking was lost, the animal left the arena or the
     experimenter's hand entered the frame.
   - An agent with tool calling runs the statistics in Python (assumption checks, then t-test, ANOVA
     or non-parametric tests) and writes a methods and results section against the ARRIVE 2.0
     checklist. Every value is traceable to the computed table, so there are no invented numbers.

**How I'll show it works.** I'll benchmark against the public dataset from Sturman et al. 2020
(*Neuropsychopharmacology*): open field videos on Zenodo with rearing annotated by three human
raters. Success means tracking that matches published tracker results, and rearing counts whose
agreement with the human raters falls within the raters' agreement with each other. Every metric
will be reported with its error, as I did for PaedScale.

**Why it matters.** Any lab with a camera gets objective, reproducible scoring for free. Animals
are scored consistently, which supports the 3Rs (reduction). The scoring logic is open and auditable
instead of a black box.

**Scope for 10 days.**
- Days 1–2: SAM 3.1 tracking and arena segmentation on open field videos.
- Days 3–4: metrics engine and unit tests.
- Days 5–6: Muse Spark behaviour classification and quality control.
- Day 7: statistics and report agent.
- Day 8: benchmark against Sturman et al.
- Days 9–10: web app, demo video, write-up.
- Elevated plus maze and forced swim are the next two tests. Multi-animal social tests are a
  stretch goal.

## Meta models / tools I'll use
- **SAM 3.1**: text-prompted video segmentation and tracking of animals; arena and zone
  segmentation; multi-object tracking for social tests.
- **Muse Spark** (via Meta Model API): video understanding for behaviour classification, tool calling
  for the analysis agent, structured output for the report.
- [Optional, if you use it] **Muse Code**: coding agent during development.

---

## Past AI project: PaedScale (main "technical skill" answer)

**What it is.** PaedScale (https://paedscale.by-dhruvik.in) is a pediatric dose-extrapolation agent
I built for the *Built with Claude: Life Sciences* hackathon (Cerebral Valley × Anthropic). Most
drugs have no pediatric guideline. PaedScale derives a defensible starting dose for a child from
adult pharmacokinetics, with citations, edge-case flags and an evidence grade from A to D. It is open
source on GitHub: https://github.com/heyiamnotacoder/dosagecalc

**Architecture.**
- **Split between AI judgment and deterministic calculation.** An orchestrator agent (Claude Opus)
  maps each drug to its elimination pathways (enzymes, transporters, active metabolites, protein
  binding) and writes the clinical justification. A deterministic Python engine does all the
  arithmetic: allometric scaling (WT^0.75) × Anderson–Holford maturation (a sigmoid Hill function of
  postmenstrual age) × organ function (bedside Schwartz eGFR). It then solves the dose for the right
  target (Css/AUC, Cmax or time>MIC).
- **Cite-or-abstain retrieval subagent** (Claude Sonnet). It searches PubMed E-utilities and openFDA
  for adult clearance and volume of distribution. If it finds no source, it returns no number and
  the case is graded D instead of the model inventing a value.
- **Safety layer.** Hard stops for contraindications, non-viable routes and toxic thresholds.
  Deterministic flags for prodrugs, obesity, high protein binding, critical illness and
  narrow-therapeutic-index drugs (TDM recommended).
- **Stack.** Python, FastAPI with a streaming endpoint, an MCP server (FastMCP) exposing the retrieval
  tools, skills loaded on demand, a single-file frontend, and deployment on Render with a custom
  domain.

**Validation.** I ran the full live agent on **60 scenarios** (20 drugs across renal, CYP and UGT
pathways × 3 clinical cases: a term neonate, a renally impaired 2-year-old and an obese 1-year-old).
Each dose was scored against a hand-anchored published pediatric guideline.
- **96%** of dosed cases (51/53) fell within the clinical 0.5–2× band.
- **60%** (32/53) fell within the strict 0.67–1.5× band.
- **82%** of cases were graded A or B.
- The engine is unit-tested separately and runs reproducibly without an API key.

**Why it's relevant here.** PharmaVision uses the same pattern: the model handles perception and
judgment, deterministic code handles every number, and the system is validated against human
ground truth with reported error.

## Techniques (if there's a separate field)
Multi-agent orchestration (orchestrator plus retrieval subagent); tool calling and MCP; retrieval
over PubMed and openFDA with abstention; separating LLM judgment from deterministic computation;
mechanistic pharmacokinetic modelling (allometry, Anderson–Holford maturation); evaluation harness
against external ground truth; FastAPI streaming; deployment.

## About me (if asked)
Junior Resident, MD Pharmacology, AIIMS Jodhpur. I build AI tools for clinical and experimental
pharmacology. My domain training covers the science (pharmacokinetics, preclinical study design,
behavioural pharmacology), and I build and validate the software myself.

## Links
- Live product: https://paedscale.by-dhruvik.in
- GitHub: https://github.com/heyiamnotacoder/dosagecalc
- LinkedIn: [add]

---

## Check before submitting
- [ ] **Eligibility:** the rules exclude "government-affiliated individuals". Confirm with the
      organisers that a resident at AIIMS (a central government institute) is eligible.
- [ ] Confirm you built PaedScale solo. If not, name your role.
- [ ] Add the firsthand line about manual behaviour scoring in your department.
- [ ] Check the Zenodo dataset loads: https://zenodo.org/record/3608658
