| Stage | Model | Seeds | Val macro-F1 | Val leaderboard F1 |
|---|---|---|---|---|
| final_candidates | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling, 20 epochs | 3 | 0.704 ± 0.006 | 0.703 ± 0.006 |
| final_candidates | ResNet-18 full fine-tune (S7 strategy 3); lr 3e-05, oversampling, 20 epochs | 3 | 0.701 ± 0.020 | 0.701 ± 0.021 |
| final_candidates | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); 20 epochs | 3 | 0.701 ± 0.014 | 0.701 ± 0.015 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 0.0001, 20 epochs | 3 | 0.700 ± 0.006 | 0.700 ± 0.006 |
| final_candidates | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling, dropout 0.3, 20 epochs | 3 | 0.698 ± 0.017 | 0.697 ± 0.014 |
| final_candidates | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); oversampling | 3 | 0.694 ± 0.025 | 0.693 ± 0.019 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); oversampling instead of class weights | 3 | 0.694 ± 0.003 | 0.688 ± 0.013 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); no class weights (control) | 3 | 0.694 ± 0.020 | 0.698 ± 0.020 |
| stage1 | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT) | 3 | 0.691 ± 0.018 | 0.694 ± 0.015 |
| sweep | ResNet-18 full fine-tune (S7 strategy 3); lr 3e-05, 20 epochs | 3 | 0.681 ± 0.017 | 0.683 ± 0.020 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 0.0003, 20 epochs | 3 | 0.679 ± 0.005 | 0.685 ± 0.007 |
| sweep | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); lr 3e-05, 20 epochs | 3 | 0.676 ± 0.011 | 0.680 ± 0.012 |
| sweep | ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT); lr 3e-05, 10 epochs | 3 | 0.674 ± 0.007 | 0.673 ± 0.019 |
| stage1 | ResNet-18 fine-tune layer3+4 (S7 mini-challenge) | 3 | 0.673 ± 0.022 | 0.678 ± 0.025 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); weight decay 1e-4 | 3 | 0.669 ± 0.021 | 0.663 ± 0.015 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); dropout 0.3 before the head | 3 | 0.668 ± 0.003 | 0.661 ± 0.017 |
| architectures | DenseNet-121 fine-tune last dense block (S7 homework) | 3 | 0.668 ± 0.015 | 0.665 ± 0.013 |
| stage1 | ViT-B/16 fine-tune last 2 blocks (S7 strategy 2 on S9 ViT) | 3 | 0.667 ± 0.019 | 0.665 ± 0.018 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); dropout 0.3 + weight decay 1e-4 | 3 | 0.666 ± 0.001 | 0.664 ± 0.006 |
| sweep | ViT-B/16 fine-tune last 2 blocks (S7 strategy 2 on S9 ViT); lr 3e-05, 10 epochs | 3 | 0.661 ± 0.014 | 0.656 ± 0.022 |
| regularization | ResNet-18 fine-tune layer3+4 (S7 mini-challenge); M1 augmentation: flips, rotation 15, colour 0.1 | 3 | 0.658 ± 0.046 | 0.660 ± 0.047 |
| stage1 | ResNet-18 fine-tune layer4 (S7 strategy 2) | 3 | 0.657 ± 0.007 | 0.664 ± 0.006 |
| stage1 | ResNet-18 full fine-tune (S7 strategy 3) | 3 | 0.651 ± 0.014 | 0.656 ± 0.014 |
| architectures | VGG-16 fine-tune 3 FC layers, dropout 0.5 (S7 slide 18) | 3 | 0.631 ± 0.006 | 0.633 ± 0.015 |
| architectures | DenseNet-121 frozen + new head (S7 homework) | 3 | 0.628 ± 0.010 | 0.632 ± 0.006 |
| stage1 | ViT-B/16 frozen + new head (S9) | 3 | 0.614 ± 0.009 | 0.612 ± 0.008 |
| architectures | EfficientNet-B0 fine-tune last 2 stages (S7 homework) | 3 | 0.606 ± 0.008 | 0.609 ± 0.013 |
| stage1 | ResNet-18 frozen + new head (S7 strategy 1, S9) | 3 | 0.595 ± 0.014 | 0.605 ± 0.011 |
| architectures | VGG-16 frozen + new head (S7 slide 18) | 3 | 0.577 ± 0.020 | 0.575 ± 0.018 |
| architectures | EfficientNet-B0 frozen + new head (S7 homework) | 3 | 0.573 ± 0.005 | 0.575 ± 0.014 |
| regularization | CNN from scratch (S7); dropout 0.3 + weight decay 1e-4 | 3 | 0.541 ± 0.008 | 0.479 ± 0.051 |
| regularization | CNN from scratch (S7); M1 augmentation | 3 | 0.539 ± 0.016 | 0.437 ± 0.103 |
| M2 baseline (features) | Random forest, 33 Session 5 features (M2) | 3 | 0.537 ± 0.001 | 0.480 ± 0.003 |
| stage1 | CNN from scratch (S7) | 3 | 0.536 ± 0.018 | 0.510 ± 0.033 |
| M2 baseline (features) | MLP 33-256-256-3 + dropout 0.3 + weight decay 1e-4 (S6) | 3 | 0.535 ± 0.004 | 0.524 ± 0.007 |
| regularization | CNN from scratch (S7); training-set mean/std normalisation | 3 | 0.525 ± 0.013 | 0.496 ± 0.018 |
| M2 baseline (features) | MLP 33-64-3, 33 S5 features (S6) | 3 | 0.490 ± 0.004 | 0.477 ± 0.007 |
| M2 baseline (features) | SVM RBF (C=1), 33 S5 features (midterm) | 3 | 0.462 ± 0.000 | 0.459 ± 0.000 |
| M2 baseline (features) | Logistic regression, 33 S5 features (S5) | 3 | 0.437 ± 0.000 | 0.408 ± 0.000 |
| M2 baseline (features) | kNN (k=15), 33 S5 features (S5) | 3 | 0.436 ± 0.000 | 0.433 ± 0.000 |
