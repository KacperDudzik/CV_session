Macro-F1 on diagnosis_1 (3 classes). Same split for every model; mean ± std over seeds; validation chooses, test scored once.

| Stage | Model | Seeds | Val macro-F1 | Test macro-F1 |
|---|---|---|---|---|
| final_candidates | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling, 20 epochs | 3 | 0.704 ± 0.006 | 0.584 ± 0.037 |
| final_candidates | ResNet-18 full fine-tune (S7 strategy 3); lr 3e-05, oversampling, 20 epochs | 3 | 0.701 ± 0.020 | 0.610 ± 0.009 |
| final_candidates | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); 20 epochs | 3 | 0.701 ± 0.014 | 0.589 ± 0.020 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 0.0001, 20 epochs | 3 | 0.700 ± 0.006 | 0.608 ± 0.016 |
| final_candidates | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling, dropout 0.3, 20 epochs | 3 | 0.698 ± 0.017 | 0.566 ± 0.056 |
| final_candidates | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); oversampling | 3 | 0.694 ± 0.025 | 0.566 ± 0.024 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling instead of class weights | 3 | 0.694 ± 0.003 | 0.608 ± 0.027 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); no class weights (control) | 3 | 0.694 ± 0.020 | 0.561 ± 0.035 |
| stage1 | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT) | 3 | 0.691 ± 0.018 | 0.595 ± 0.039 |
| sweep | ResNet-18 full fine-tune (S7 strategy 3); lr 3e-05, 20 epochs | 3 | 0.681 ± 0.017 | 0.612 ± 0.017 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 0.0003, 20 epochs | 3 | 0.679 ± 0.005 | 0.567 ± 0.020 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 3e-05, 20 epochs | 3 | 0.676 ± 0.011 | 0.613 ± 0.023 |
| sweep | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); lr 3e-05, 10 epochs | 3 | 0.674 ± 0.007 | 0.579 ± 0.029 |
| stage1 | ResNet-18 fine-tune layer3+4 (S7 mini-challenge) | 3 | 0.673 ± 0.022 | 0.572 ± 0.009 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); weight decay 1e-4 | 3 | 0.669 ± 0.021 | 0.598 ± 0.045 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); dropout 0.3 before the head | 3 | 0.668 ± 0.003 | 0.588 ± 0.019 |
| architectures | DenseNet-121 fine-tune last dense block (S7 homework) | 3 | 0.668 ± 0.015 | 0.578 ± 0.014 |
| stage1 | ViT-B/16 fine-tune last 2 blocks (S7 strategy 2 on S9 ViT) | 3 | 0.667 ± 0.019 | 0.566 ± 0.010 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); dropout 0.3 + weight decay 1e-4 | 3 | 0.666 ± 0.001 | 0.586 ± 0.051 |
| sweep | ViT-B/16 fine-tune last 2 blocks (S7 strategy 2 on S9 ViT); lr 3e-05, 10 epochs | 3 | 0.661 ± 0.014 | 0.577 ± 0.041 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); M1 augmentation: flips, rotation 15, colour 0.1 | 3 | 0.658 ± 0.046 | 0.570 ± 0.016 |
| stage1 | ResNet-18 fine-tune layer4 (S7 strategy 2) | 3 | 0.657 ± 0.007 | 0.583 ± 0.023 |
| stage1 | ResNet-18 full fine-tune (S7 strategy 3) | 3 | 0.651 ± 0.014 | 0.582 ± 0.017 |
| architectures | VGG-16 fine-tune 3 FC layers, dropout 0.5 (S7 slide 18) | 3 | 0.631 ± 0.006 | 0.571 ± 0.019 |
| architectures | DenseNet-121 frozen + new head (S7 homework) | 3 | 0.628 ± 0.010 | 0.546 ± 0.002 |
| stage1 | ViT-B/16 frozen + new head (S9) | 3 | 0.614 ± 0.009 | 0.589 ± 0.021 |
| architectures | EfficientNet-B0 fine-tune last 2 stages (S7 homework) | 3 | 0.606 ± 0.008 | 0.576 ± 0.008 |
| stage1 | ResNet-18 frozen + new head (S7 strategy 1, S9) | 3 | 0.595 ± 0.014 | 0.580 ± 0.026 |
| architectures | VGG-16 frozen + new head (S7 slide 18) | 3 | 0.577 ± 0.020 | 0.542 ± 0.016 |
| architectures | EfficientNet-B0 frozen + new head (S7 homework) | 3 | 0.573 ± 0.005 | 0.593 ± 0.013 |
| regularization | CNN from scratch (S7); dropout 0.3 + weight decay 1e-4 | 3 | 0.541 ± 0.008 | 0.514 ± 0.036 |
| regularization | CNN from scratch (S7); M1 augmentation | 3 | 0.539 ± 0.016 | 0.539 ± 0.023 |
| M2 baseline (features) | Random forest, 33 Session 5 features (M2) | 3 | 0.537 ± 0.001 | 0.511 ± 0.023 |
| stage1 | CNN from scratch (S7) | 3 | 0.536 ± 0.018 | 0.540 ± 0.011 |
| M2 baseline (features) | MLP 33-256-256-3 + dropout 0.3 + weight decay 1e-4 (S6) | 3 | 0.535 ± 0.004 | 0.525 ± 0.034 |
| regularization | CNN from scratch (S7); training-set mean/std normalisation | 3 | 0.525 ± 0.013 | 0.489 ± 0.013 |
| M2 baseline (features) | MLP 33-64-3, 33 S5 features (S6) | 3 | 0.490 ± 0.004 | 0.513 ± 0.015 |
| M2 baseline (features) | SVM RBF (C=1), 33 S5 features (midterm) | 3 | 0.462 ± 0.000 | 0.479 ± 0.004 |
| M2 baseline (features) | Logistic regression, 33 S5 features (S5) | 3 | 0.437 ± 0.000 | 0.464 ± 0.000 |
| M2 baseline (features) | kNN (k=15), 33 S5 features (S5) | 3 | 0.436 ± 0.000 | 0.449 ± 0.000 |
| final | **Best → final: ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling, 20 epochs, seed 2** | 1 | **0.710** | **0.618** |
