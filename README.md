# MILK10k lesion classification: Computer Vision course project (EADA)

An individual project: a full classification pipeline for MILK10k skin lesions, built milestone by
milestone through the course, ending with an official ISIC leaderboard submission.

| Milestone | Sessions | Where |
|---|---|---|
| M1 Data pipeline: EDA, quality, lesion-level splits | S1–S3 | [s1_s3_homework/](s1_s3_homework/) |
| M2 Classical baseline: hand-crafted features + classical models | S4–S5 | [ML_midterm_project/](ML_midterm_project/), rerun on the final split in [final_project/baselines.py](final_project/baselines.py) |
| M3 Deep learning: MLP, CNN from scratch, ResNet-18 transfer learning | S6–S7 | [session6/](session6/), [session7/](session7/) |
| M4 Best model, error analysis, fairness | S8–S9 | [final_project/](final_project/) (`train.py`, `evaluate.py`, `analysis.py`) |
| M5 Final pipeline, 11-class model, leaderboard | S10 | [final_project/](final_project/) (`predict.py`, `make_submission.py`, `submission/`) |

**Start here:** [final_project/README.md](final_project/README.md) (how to run everything, and the
rules every experiment follows), [final_project/report.md](final_project/report.md) (final report),
[final_project/model_card.md](final_project/model_card.md).

Data: MILK study team. MILK10k Benchmark. ISIC Archive, 2025. doi:10.34970/262082.
Tschandl P, et al. J Invest Dermatol. 2026;146(2):357-364. doi:10.1016/j.jid.2025.06.1594.
Licence CC-BY-NC (non-commercial use only). Images are not committed; place them in `milk10k/`
or set `MILK10K_DIR`.
