# Fusion H6a release assets

## Final neural networks and associated inputs

| Asset | Exact file or identity | Source and purpose |
| --- | --- | --- |
| A1 detector | `best_detector.pth`, epoch 3, SHA-256 `c32dc1ffd885f0f076e840a84b3edd40632cbc3eaf55ca949c8f5ecb2328334f` | [A1 dataset V1](https://www.kaggle.com/datasets/zhuowamg/a1data/versions/1). [Training, architecture and loader](../notebooks/02_a1_detector.ipynb). Required upstream for Mul2, along with the archive's run receipt and quality-model records. |
| V1 U-Net refiner | `checkpoints/refiner_best.pt` and the original normalization/configuration. The signed A1 `baseline_audit.json` supplies expected `v1_checkpoint_sha256`. | [V1 dataset V1](https://www.kaggle.com/datasets/zhuowamg/v1output/versions/1). [Training and architecture](../notebooks/01_v1_training.ipynb). The V1 detector also supplies initialization for A1 training. |
| S0 foreground ensemble | Five ResNet-34 and five EfficientNetV2-S models. [Ten filenames and SHA-256 digests](../evidence/f4_input_identities.json). Original `solar_unet_receipt.json` and `folds.csv`. | [Author's V2 dataset](https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights/versions/2). [Preprocessing, model loading and 8-view inference](../src/h6a/probability.py). Automatically accessed by the demo. CUDA inference checks every model digest before loading. |
| Mul2 upstream GitHub U-Net | `validated_unet_run3_last.pt`, SHA-256 `1d1dfd8042b44a198d52efb4c5b928abbf861cbe0e973a1ee77174ad19bf06f9`. | [Weight dataset](https://www.kaggle.com/datasets/ameypawar123456789/filament-seg-weights) and [original model code](https://github.com/ameyypawar/filament-seg). The original Mul2 loader validates bytes against the G3 asset contract. |
| Mul2 upstream Blended U-Net | `solar_v3/blended_shared_split_last.pt`. The original Mul2 loader also validates this against the G3 asset contract. | [V3 weight dataset](https://www.kaggle.com/datasets/zhuowamg/v3output), [training workflow](../notebooks/06_blended_training.ipynb), and [public architecture reference](https://www.kaggle.com/code/anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments). |
| H6a quality estimator and policy | [Frozen Ridge coefficients, preprocessing constants and decision price](../src/h6a/fusion_reference.json). | Load from Git. [F4 contour rules](../src/h6a/rules.py), [candidate features/scoring](../src/h6a/candidate.py) and [H6a selection](../src/h6a/fusion_policy.py) run on the selected input predictions. |

The two Mul2 semantic models are separate from the ten-model S0 ensemble. For sources without a version established in the release evidence, the table links the dataset and specifies a component identity or an existing content check. Model assets stay at their original sources. The release neither changes their applicable terms nor republishes binaries. See [source acknowledgments](upstream_attribution.md) and the upstream links for terms. Training optimizer state and intermediate epoch checkpoints are not used by the demo.

## Intermediate predictions and how they are used

| Asset | Role |
| --- | --- |
| [Signed Mul2 receipts V1](https://www.kaggle.com/datasets/zhuowamg/mul2out/versions/1) | Native-grid four-view masks and scores, `vote_both` masks and per-photo image and receipt identity. Automatically acquired in both executable demo modes. |
| [Signed S0 maps V2](https://www.kaggle.com/datasets/lojhhggg/sanxiandata/versions/2) | Original test maps. Automatically acquired by replay. The GPU `fresh_s0` mode regenerates these maps from model forwards instead. The historical archive also contains data used by notebook 05. |
| [G3 V1](https://www.kaggle.com/datasets/wanhty/g-data/versions/1), [Q1 V2](https://www.kaggle.com/datasets/lojhhggg/v4output/versions/2), [prior contour V1](https://www.kaggle.com/datasets/zhuowamg/a1mulout/versions/1) | The historic Mul2 test launcher reads normalized/semantic G/Q intermediates, original A1 seeds from the A1 archive, and optionally the original contour masks. It runs the four augmented detector/refiner passes. This is the remaining dependency in the raw-image path. |
| [Historical OOF probability cache V1](https://www.kaggle.com/datasets/llooppoo/errordata/versions/1) | Optional historical evaluation in notebook 05. The demo never opens it or any ground-truth labels. |
| H0/H1/H2 reference CSVs and archived H6a CSV | Historical identity checks and scoring evidence. The demo neither opens nor copies them. It generates F4 masks, features, quality scores, H6a choices, and CSV rows from the inputs. |

The official test images remain on Kaggle. Join the competition and attach its official dataset to the runtime. The launcher uses KaggleHub's versioned dataset resolver for its public model and cache assets. A Kaggle error or a hash mismatch ends the run with a specific error. No personal token is required in the source.
