# Cross-Signer ASL Word Recognition on WLASL

This project investigates cross-signer generalization for isolated American Sign Language word classification on the WLASL-100 subset. The main question is whether a skeleton-only pose pipeline can recognize ASL glosses from unseen signers when RGB appearance cues such as skin tone, clothing, background, and camera setup are removed.

The pipeline uses MediaPipe Holistic to extract 2D body and hand landmarks, applies SPOTER-style signing-space and per-hand normalization, and trains a Transformer-based classifier on normalized pose sequences. A signer-disjoint train/validation/test split is constructed to ensure that no signer appears in more than one partition.

The project evaluates the model using top-1 accuracy and macro-F1 on a held-out signer-disjoint test set. It also includes a Park-inspired preprocessing extension with palm-anchor translation and bilinear hand-keypoint reconstruction to test whether stronger hand normalization improves generalization.

The results show that normalized skeleton sequences contain transferable gloss information and perform above chance, but the large train–test gap indicates that pose normalization alone is not sufficient for robust cross-signer recognition. Failure analysis highlights two main limitations: signer-specific kinematic shift and fine-grained 2D pose collapse caused by missing depth and hand-contact cues.
