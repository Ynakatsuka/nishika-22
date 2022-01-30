#!/bin/bash
python run.py "$@"
python run.py mode=inference "$@"
python src/misc/make_submission.py "$@"
