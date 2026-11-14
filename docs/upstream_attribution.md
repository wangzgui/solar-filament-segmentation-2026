# Source acknowledgments

The Mask R-CNN v2 architecture, anchors, ROI mask pooling, SGD/DDP/AMP training design, ResNet18 U-Net decoder and crop geometry are based on Nomannic19's Apache-2.0 Kaggle notebook, supplied locally as `cv-solar-filament-mask-r-cnn-u-net-refiner.ipynb` (35 cells). `reference_models.py` retains the original architecture definitions. V1 changes the data handling, preprocessing, refiner input, losses, checkpoint selection, postprocessing, calibration and orchestration.

Original: https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner

`disk.py` is adapted from ameyypawar/filament-seg, commit 81a739bb7786a44b989f5d76afbb6f9d22b5d1dd, under the MIT license. The original MIT license is included as GITHUB_REFERENCE_LICENSE. The V1 radial profile uses edge-padded smoothing to avoid convolution boundary artifacts.

Reference: https://github.com/ameyypawar/filament-seg

The official-compatible scorer preserves the all-pairs, strict IoU > 0.5 and pooled annotator semantics of the host Self_Evaluation_Notebook, version 6. Tests compare its counts and PQ with functions extracted from that public source.

Official: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

BCE + Tversky and crop-level segmentation are informed by the locally supplied MAGFiLO notebook. Its heavy YOLO architecture is not used. Artem Shelamanov's public portfolio informs the ensemble/quality interface only; its unpublished code is not claimed to have been reproduced.

https://www.shelamanov.com/portfolio/solar
