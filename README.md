# Talking Head Anime

Reproducible implementation
for [Talking Head Anime From a Single Image](https://pkhungurn.github.io/talking-head-anime/) .

Currently supports:

* Processing 3D Models to ***Talking Head Anime*** trainable dataset.
* Training ***Talking Head Anime***.
* ***Talking Head Anime*** inference.

## Environment

Supports `Docker` and `Conda` environment.

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

### Training Configs

Trainer configs are at `train_morpher.yaml`, `train_rotator.yaml` .

#### General train configs

General training configs are at `logging` part from `train_*.yaml` file.

```yaml
logging:
  log_dir: "./logs/"
  seed: "16" # use your seed for each training
  nepochs: 10000 # maximum epochs
  device: cuda
  save_optimizer_state: False

  freq: 500 # logging frequency(step)

  save_files: [
      '*.py',
      '*.sh',
      'configs/*.*',
      'configs/dataset/*.*',
      'datasets/*.*',
      'models/*.*',
      'utils/*.*',
  ]
```

#### Model configs

Model configs are at `models` part from `train_*.yaml` file.

* Change model class if needed.
* Change optimizer class, lr and betas if needed.

```yaml
models:
  FaceMorpher:
    class: models.tha1.FaceMorpher

    optim:
      class: torch.optim.Adam
      kwargs:
        lr: 1e-4
        betas: [ 0.5, 0.999 ]
```

#### Dataset configs

1. Each dataset should have corresponding config `.yaml` file.
2. Each corresponding `.yaml` file should be in `datasets.*.datasets` list in general train config file.

```yaml
datasets:
  train: # configs for training dataset
    class: datasets.base.MultiDataset
    datasets: [
        'configs/datasets/custom.yaml', # add path to your dataset config file here
    ]

    mode: train
    batch_size: 25
    shuffle: True
    num_workers: 8

  eval: # configs for eval dataset
    class: datasets.base.MultiDataset
    datasets: [
        'configs/datasets/custom.yaml', # add path to your dataset config file here
    ]

    mode: eval
    batch_size: 25
    shuffle: False
    num_workers: 2
```

### Tensorboard

Training logs are logged with tensorboard.

Run `tensorboard --logdir ./logs/<YOUR SEED> --bind_all` to check logs.

## Inference

Check `Inference.ipynb`.

## License

NEEDS WORK

## Author

* [Dongho Choi](https://github.com/dhchoi99)
* [jhr337](https://github.com/jhr337)

Special thanks to:

* [MINDsLab Inc.](https://github.com/mindslab-ai) for GPU support

## References

* [Talking Head Anime from a Single Image](https://pkhungurn.github.io/talking-head-anime/)

## TODO

* Combiner not trained yet...
* Code cleanup