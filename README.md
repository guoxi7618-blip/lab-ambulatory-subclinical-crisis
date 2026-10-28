# Laboratory stress regulation and ambulatory assessment of subclinical psychological crisis

Code, analysis outputs and de-identified processed data for:

> Li, Y., Wang, X., Xing, Y., He, J., Yang, S., Cui, J., & Dong, Y. *Laboratory Stress Regulation and Ambulatory Assessment of Subclinical Psychological Crisis in University Students.* Manuscript submitted to *Psychophysiology*.

Two semester cohorts of university students (core/development, n = 203; extension/temporal validation, n = 198) completed a counselor evaluation, a laboratory stress task (Montreal Imaging Stress Task) with 32-channel EEG, ECG and palmar EDA, and 30 days of wrist monitoring (Empatica EmbracePlus) with ecological momentary assessment (EMA).

## Repository contents

| Path | Content |
|---|---|
| `data/` | De-identified processed data (participant, laboratory, ambulatory, analysis-ready and model-prediction tables), source data for tables and figures, and `10_codebook/codebook.csv` |
| `audit_20261028_*.py` | Statistical analyses: H1 laboratory markers and replication, H2 laboratory–field models, H3 identification metrics, paired DeLong contrasts, elastic-net baselines, calibration sensitivity, daily reference frames |
| `revision_20261028_h4.py` | H4 ecological performance: scenario strata, examination weeks, GEE and within-person estimands |
| `revision_20261028_run_experiments.py` | Entry point listing and running the analysis stages |
| `audit_20261028_*.csv/json/txt`, `revision_20261028_h4_*` | Outputs of the analyses as reported in the manuscript |
| `revision_20261028_figures_v4_panels.py` | Data panels for all figures (matplotlib, MNE-Python topographies), computed from `data/` |
| `revision_20261028_figures_v4_pptx.py`, `export_figures_v4.ps1` | Figure assembly in PowerPoint (native shapes) and export to PDF/600-dpi PNG |
| `Figures/v4/final/` | Final figures (PDF); `Figures/v4/panel_numbers_v4.json` lists the values checked against the manuscript |

## Data availability and de-identification

Processed (feature-level) data are released here. Raw recordings (EEG, ECG, EDA and wrist signals) and the restricted variables listed below are available from the corresponding author (Yindong Dong, dongyindong66@163.com) on reasonable request, subject to participant privacy and institutional requirements.

Restricted variables (removed from this release; marked in the `public_release` column of the codebook):

- Counselor-rated crisis-signal indicators and all C-SSRS suicide-risk items (`reference_standard.csv`)
- Passive suicidal ideation in EMA diaries (`ema_prompts.csv`, `prompt_level_h2.csv`) and its 30-day summary (`person_level_features.csv`)
- Day 31 referral status, academic discipline, year of study, height, weight (BMI retained) and menstrual-cycle information

Participants are identified only by study pseudonyms (C = core, E = extension). Calendar dates embedded in EMA prompt IDs (and in `revision_20261028_h4_dates.csv`) were shifted by a random whole number of weeks per participant; weekdays, study days and within-person intervals are preserved. `release_log.json` lists every change.

## Reproducing the analyses

Python 3.11 or later; install dependencies with

```bash
pip install -r requirements.txt
```

List and run the analysis stages from the repository root:

```bash
python revision_20261028_run_experiments.py --list
python revision_20261028_run_experiments.py --tag reproduce
```

Regenerate the figure panels and the PowerPoint figures (export requires Microsoft PowerPoint on Windows):

```bash
python revision_20261028_figures_v4_panels.py
python revision_20261028_figures_v4_pptx.py
powershell -ExecutionPolicy Bypass -File export_figures_v4.ps1
```

Scope of reproduction with the public data:

- Laboratory markers, EEG topographies, laboratory–field models, all identification metrics and paired contrasts (from the archived prediction files), H4 ecological analyses and within-person discrimination run directly on the released data.
- Steps that use restricted variables need the restricted files: the data-integrity checks (`audit_20261028_integrity*.py`, which also verify input hashes of the original files), the crisis-presentation subgroup AUCs and the elastic-net models whose EMA block includes the passive-ideation summary (`audit_20261028_models.py`), and `audit_20261028_supplementary_audit*.py`. Their outputs are included in this repository; the figure script plots the crisis-presentation panel from `Figures/v4/panel_numbers_v4.json` when the restricted columns are absent.
- The original multiscale attention-based fusion (MSAF) network is evaluated through its archived extension-cohort probabilities (`data/07_model/`); its five-minute training tensors and weights are not part of this release.

## License

Code: MIT License (`LICENSE`). Data and figures: Creative Commons Attribution 4.0 International (CC BY 4.0).
