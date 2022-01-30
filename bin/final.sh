#!/bin/bash
# ---------------------------
# 1. preprocess
# ---------------------------
python src/misc/preprocess.py

# ---------------------------
# 2. model
# ---------------------------
./bin/tools/train_and_inference.sh experiment=exp012 model.model.backbone.name=swin_base_patch4_window7_224
./bin/tools/train_and_inference.sh experiment=exp016 model.model.backbone.name=convnext_large_in22ft1k
./bin/tools/train_and_inference.sh experiment=exp032

# ---------------------------
# 3. Ensemble
# ---------------------------
python src/misc/make_ensemble_submission.py
