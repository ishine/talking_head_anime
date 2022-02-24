# Talking Head Anime

Repository to make [Talking Head Anime](https://pkhungurn.github.io/talking-head-anime/) reproducible.

Currently supports:

* Processing 3D Models to ***Talking Head Anime*** trainable dataset.
* Training ***Talking Head Anime***.
* ***Talking Head Anime*** inference.

## Environment

See [`Environment.md`](./Environment.md).

## Dataset

You can train Talking Head Anime with two different type of datasets:

1. Images Dataset (recommended)
2. 3D-models Dataset

Check `dataset.ipynb` for details. You can generate your own dataset when following `dataset.ipynb`.

## Training

### Morpher

`python train_morpher.py --train`

### Rotator

`python train_rotator.py --train`

### Combiner

TODO

## Inference

Check `Inference.ipynb`.

Inference currently only works with GUI-available environment.


## TODO

* Combiner not trained yet...
* Dockerfile
* Code cleanup