#!/bin/bash
args=$(echo "$@" | sed 's/ /_/g' | sed 's/=/_/g')
rnd=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1 | sort | uniq)
grp_name=$args_$rnd

python run.py "$@" trainer.logger.group=$grp_name
python run.py "$@" trainer.logger.group=$grp_name trainer.idx_fold=1
python run.py "$@" trainer.logger.group=$grp_name trainer.idx_fold=2
python run.py "$@" trainer.logger.group=$grp_name trainer.idx_fold=3
python run.py "$@" trainer.logger.group=$grp_name trainer.idx_fold=4
python run.py "$@" trainer.logger.group=$grp_name trainer.idx_fold=5
python run.py "$@" trainer.logger.group=$grp_name run=evaluate_oof
