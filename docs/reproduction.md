# Reproducing Fusion H6a

## Start with the evaluator notebook

1. Import [00_final_demo.ipynb](../notebooks/00_final_demo.ipynb) into Kaggle. Join the competition and attach the [official competition dataset](https://www.kaggle.com/competitions/filament-segmentation-2026/data). Select GPU (dual T4 recommended) and enable Internet so the notebook can clone the public source and install missing packages.
2. In the first cell use `MODE = "full_inference"`. Set `SMOKE_PHOTOS = 2` for a short interactive check. KaggleHub resolves the model assets and recipe manifests. Do this interactively before using Save Version so Kaggle can attach the sources. Smoke output has its own filename and is not a competition submission. The post-inference `smoke_comparison.json` compares those photos with the archived mask sequences.
3. For the all-photo run set `SMOKE_PHOTOS = 0`. A matching result is written under `/kaggle/working/solar_final_demo/full_inference-*/submission.csv`. The signed CSV is accepted only after mask validation and a byte-for-byte SHA-256 check. Preserve the run manifest if there is a mismatch.

| Mode | Starts from | Verification |
| --- | --- | --- |
| `full_inference` (default, CUDA) | Official test images, signed A1/V1, two Mul2 semantic checkpoints, ten S0 checkpoints, frozen HGB/Ridge models and recipe manifests. Generates A1 seeds, four transformed proposal streams, Mul2 contour/vote and S0 before H6a. | Launcher and frozen handoff are tested with synthetic data. Real Kaggle GPU execution and byte parity are still pending. |
| `verified_replay` (CPU) | Original signed Mul2 and S0 predictions; F4 masks, candidate scores, H6a selection and CSV are recomputed | Original notebook 05 produced the champion bytes on Kaggle. The streamlined launcher retains its local checks. |
| `fresh_s0` (CUDA) | Fresh S0 neural forwards and archived Mul2 receipts | Useful for isolating S0 parity. Real GPU parity is still pending. |

The original Mul2 archive export read A1 predictions and G/Q cached arrays. The new raw-image loader uses the exact original Mul2 source in notebook 03, validates G/Q metadata and checkpoints, and runs the original A1 main-view, two flips, two rotations and contour algorithms fresh. It never loads G/Q photo predictions, archived A1 seeds, archived contours, Mul2/S0 caches or test labels. It reads signed G/Q configuration manifests and the frozen upstream C-HGB model. Dataset mounts may include other files that the raw loader does not read.

Use the recorded Tesla T4 / CUDA 12.8, PyTorch 2.10.0+cu128, TorchVision 0.25.0+cu128 and OpenCV 4.13.0 versions when available. The launch records installed versions. Budget at least 8 GB for S0 temporary arrays, plus model, map and scratch storage. CUDA/kernel differences can affect byte parity. A full GPU run from an evaluator-accessible account is still the release acceptance test.

The demo's default run skips training and evaluation. Its optional section points to the original training notebooks, model configuration, and labeled development/audit path. The [asset inventory](assets.md) identifies neural checkpoints, cached predictions, and files generated at run time.

## Test-only inference from intermediate predictions

Use the official test images, signed Mul2 receipts V1, the Afshar V2 ensemble receipt, and the signed S0 test probability maps V2, linked below. Mount them in Kaggle with this repository checkout and run:

```bash
python src/h6a/predict_champion.py --input-root /kaggle/input --output /kaggle/working/sub_H6a.csv
```

This path computes the F4 associations directly from the candidate masks and semantic connected components. It rebuilds F4 contours, computes candidate features and Ridge scores, then runs the H6a mixed-integer selection and serializes new masks. It reads frozen Ridge parameters, original Mul2 proposal receipts, and original S0 probability maps. It does not load a reference submission, frozen per-photo F4 matches, historical probability maps, or ground-truth annotations. The expected champion digest is consulted only after serialization. The original Kaggle run below is the recorded byte-identical execution. This streamlined path has local synthetic equivalence coverage and is pending execution on the complete mounted Test set.

## Verified cached replay

1. Import [05_h6a_fusion.ipynb](../notebooks/05_h6a_fusion.ipynb) into Kaggle and add the five inputs below. Preserve the archive layouts.
2. Run the single cell on CPU. The notebook recreates semantic-guided F4 masks, rebuilds the candidate bank and candidate features, replays the frozen quality estimators and MILP, and writes `sub_H6a.csv`. The audit notebook also evaluates two annotated historical cohorts. Test annotations are neither required nor used.
3. Find `/kaggle/working/F4_Candidate_Fusion_V1/run_*/sub_H6a.csv`. Run `python verify_submission.py /path/to/sub_H6a.csv` to check the masks and the expected SHA-256 `0d170beedd870d5375cb83c3c83d1bc57a3dd852fd715be5b5bec9292a6e5dee`. Select H6a explicitly. The original notebook names its development winner independently.

| Kaggle input | Required files |
| --- | --- |
| [Competition data](https://www.kaggle.com/competitions/filament-segmentation-2026/data) | Official train annotation JSON with SHA-256 `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`, plus official train/test images adjacent to the JSON under the standard layout. |
| [Mul2 receipts V1](https://www.kaggle.com/datasets/zhuowamg/mul2out/versions/1) | `submissionlast.csv`, run and submission identity JSON, validation roster, per-photo summaries and `photo_receipts/{selection,audit,test}`. The original `vote_both` CSV SHA-256 is `ba78fdd88a08ec280197f4e2120a9abf850798de4983195c11f80f9c5fd795dd`. Each candidate receipt is verified against its source image and stored content. |
| [Afshar ensemble V2](https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights/versions/2) | Ten weight files, the author's receipt, and `folds.csv`. See the [individual checkpoint hashes](../evidence/f4_input_identities.json). The cached CPU run also checks these assets. |
| [S0 cache V2](https://www.kaggle.com/datasets/lojhhggg/sanxiandata/versions/2) | Test and historical `baseline_u8` arrays under `Solar_S1_0/p039_cache/4e7786a7260fb1c37_d850879ac830/`. |
| [Historical probability cache V1](https://www.kaggle.com/datasets/llooppoo/errordata/versions/1) | Remaining historical predictions. The [per-photo path manifest](../src/h6a/reference_cache.json) identifies the exact out-of-fold maps across both archives. |

The [frozen decision assets](../src/h6a/fusion_reference.json) include scoring coefficients, training dates and control decisions. H0-H2 reference CSVs are used for identity checks. H6a is calculated from candidate masks and written separately.

The original Kaggle Fusion run completed the signed replay. Its [status](../evidence/h6a_kaggle_status.json), [submission manifest](../evidence/h6a_kaggle_submission_manifest.json), and [Selection/Audit metrics](../evidence/h6a_selection_audit.csv) are archived here.

## Producing the upstream inputs

These notebooks document the stages used to create the signed inputs. Their use with newly regenerated assets is distinct from replaying the exact published cache. The historical main-view pipeline was independently replayed on 35 photos. End-to-end GPU regeneration of all 180 test photos and equality with the signed H6a CSV have not been independently verified in this release.

| Stage | Executable notebook | Inputs and outputs |
| --- | --- | --- |
| V1 instance training | [01_v1_training](../notebooks/01_v1_training.ipynb) | Official training annotations, COCO detector / ImageNet encoder initialization, date grouping, disc normalization, detector and ROI refiner. The signed [V1 output V1](https://www.kaggle.com/datasets/zhuowamg/v1output/versions/1) contains the original checkpoints and normalization. |
| A1 detector adaptation | [02_a1_detector](../notebooks/02_a1_detector.ipynb) | Official data, V1 weights, [G3 V1](https://www.kaggle.com/datasets/wanhty/g-data/versions/1) and [Q1 V2](https://www.kaggle.com/datasets/lojhhggg/v4output/versions/2). Load the archived A1 epoch-3 checkpoint from [A1 V1](https://www.kaggle.com/datasets/zhuowamg/a1data/versions/1). |
| Four-view proposals and voting | [03_mul2_multiview](../notebooks/03_mul2_multiview.ipynb) | Uses the A1, V1, G3 and Q1 assets above, [original A1 contour reference](https://www.kaggle.com/datasets/zhuowamg/a1mulout/versions/1), [GitHub U-Net weights](https://www.kaggle.com/datasets/ameypawar123456789/filament-seg-weights), and [Blended weights](https://www.kaggle.com/datasets/zhuowamg/v3output). Writes the cached `vote_both` and four geometrically aligned donor streams. |
| S0 semantic probability | [04_s0_semantic](../notebooks/04_s0_semantic.ipynb) | Official data and the Afshar V2 ensemble checkpoint receipt. Five ResNet34 and five EfficientNetV2-S U-Nets at native resolution with eight D4 views. The cached `baseline_u8` path is used by H6a. |
| Semantic correction and instance-set selection | [05_h6a_fusion](../notebooks/05_h6a_fusion.ipynb) | Uses the signed inputs listed above. The stored Ridge weights are used for test predictions. |

The historical [G3 graph](../notebooks/07_g3_graph.ipynb) and [Q1 quality](../notebooks/08_q1_quality.ipynb) notebooks document the two A1 upstream cache producers. They use the signed V1 and semantic assets above and the [legacy V1.4 archive](https://www.kaggle.com/models/zhuowamg/v1-4output) for provenance checks. Replaying Fusion H6a can start directly from the mounted signed Mul2 and S0 inputs.

The recorded main-view replay ran on a Tesla T4 with CUDA 12.8, PyTorch 2.10.0+cu128, TorchVision 0.25.0+cu128 and OpenCV 4.13.0. `requirements.txt` pins Python package releases. Keep Kaggle's matching CUDA builds for the GPU stages. The lightweight H6a replay and CSV validator can use CPU.

If a Kaggle link requires access, the missing input must be made available to the evaluator before independent reproduction can be considered complete. A reference CSV alone cannot substitute for split membership, candidate receipts, or model assets.

The optional historical Blended training notebook uses its original missing-package fallback for `pycocotools==2.0.10`. Fusion replay uses `pycocotools==2.0.11`.

## Model sources and exact upstream weights

- **A1 detector.** Use `best_detector.pth` (epoch 3) from the A1 V1 archive. Its SHA-256 is `c32dc1ffd885f0f076e840a84b3edd40632cbc3eaf55ca949c8f5ecb2328334f`. The A1 notebook contains the training, architecture and inference code.
- **V1 refiner.** Use `checkpoints/refiner_best.pt` and the normalization assets from the V1 archive. Their hashes must match `v1_checkpoint_sha256` in A1's signed `baseline_audit.json`. See the V1 training notebook and the [original architecture source](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner).
- **Mul2 semantic inputs.** The [GitHub U-Net source](https://github.com/ameyypawar/filament-seg) corresponds to `validated_unet_run3_last.pt` (SHA-256 `1d1dfd8042b44a198d52efb4c5b928abbf861cbe0e973a1ee77174ad19bf06f9`). The Blended checkpoint is `solar_v3/blended_shared_split_last.pt`. Mul2 validates both against the G3 asset contract and embeds their inference code. The [V3 training notebook](../notebooks/06_blended_training.ipynb) contains the Blended adaptation.
- **H6a S0 ensemble.** The author describes the method in the linked checkpoint archive. The [semantic notebook](../notebooks/04_s0_semantic.ipynb) contains the eight-view inference code that generates the probability maps used by H6a.
