# Solar Filament Segmentation Challenge 2026

Solution by Zhuoer Wang. The archived Fusion H6a submission displayed **0.40 PQ** on the Kaggle public leaderboard. The historical 16-photo development and 19-photo audit scores were 0.465532 and 0.475113.

The pipeline combines Mask R-CNN proposals, an instance-conditioned U-Net refiner, four-view mask aggregation, an independent semantic ensemble, and constrained selection of non-overlapping masks.

- [Technical report, four-page two-column edition](report/technical_report.pdf)
- [Organizer-template report edition](report/technical_report_acm.pdf)
- [Start here: evaluator demo](notebooks/00_final_demo.ipynb)
- [Versioned models, weights and intermediate files](docs/assets.md)
- [Full pipeline and input instructions](docs/reproduction.md)
- [Exact model checkpoints and original source links](docs/reproduction.md#model-sources-and-exact-upstream-weights)
- [Final Fusion notebook](notebooks/05_h6a_fusion.ipynb)
- [Test-only H6a inference](src/h6a/predict_champion.py)
- [Readable Fusion source](src/h6a)
- [Signed submission](results/submission.csv)
- [Acknowledgments and original references](docs/upstream_attribution.md)

The evaluator notebook defaults to full GPU inference from official test images and signed, automatically resolved assets. It rebuilds both neural branches and H6a without training or archived predictions. Use a short smoke run first. The new raw-image path has synthetic integration coverage and awaits a real Kaggle run and championship checksum. The original cached replay remains available as a CPU mode. The validator checks the signed file's SHA-256, RLE format and pixel exclusivity.

Competition images and ground-truth annotations are obtained from the official competition. The repository contains code and aggregate diagnostic evidence. Checkpoint and cache locations are documented in the reproduction guide.

To rebuild the organizer-template edition from its LaTeX source, run `tectonic main.tex` inside `report/`, or compile the same source and figures in Overleaf using the organizer's ACM template.
