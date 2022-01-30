#!/bin/bash
python run.py run=inference "$@"
python run.py run=inference "$@" trainer.idx_fold=1
python run.py run=inference "$@" trainer.idx_fold=2
python run.py run=inference "$@" trainer.idx_fold=3
python run.py run=inference "$@" trainer.idx_fold=4
