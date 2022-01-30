<div align="center">

# X place solution of "AI× 商標：イメージサーチコンペティション（類似商標画像の検出）"

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Nanika Kaku

## Setup

```
COMPETITION_NAME='nishika-22'

git clone https://github.com/Ynakatsuka/$COMPETITION_NAME
cd $COMPETITION_NAME

# Credentials
cp .env.template .env
# vim .env  # Set your credentials

# Download data & unzip to `data/input/`

# Delete empty directories
find data/input/cite_images/ -empty -type d -delete
```

If you don't use docker, you need to add PYTHONPATH

```
export PYTHONPATH="${PWD}/src:$PYTHONPATH"
```

## How to Run

- Preprocess and train models and inference

```
docker-compose up -d
docker-compose exec kaggle ./bin/final.sh
```

- Benchmark solution

```
docker-compose exec kaggle python src/benchmark/benchmark.py
```

- Demo App

```
docker-compose exec kaggle streamlit run src/benchmark/app.py
```

Then, access here: http://localhost:8501/

## Others

- Format code

```
pysen run lint && pysen run format
```

## References

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [kvt](https://github.com/pudae/kaggle-understanding-clouds)
