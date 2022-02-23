## Environment

See [`Environment.md`](./Environment.md).

```shell
ln -s YOUR_DATA_PATH data
ln -s YOUR_BLENDER_PATH blender
ln -s YOUR_LOG_PATH logs
```

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

Check `Inference.ipynb'`.

Inference currently only works on with-GUI environment.