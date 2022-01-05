## Environment

use conda env

### Installing bpy

Basics:

```shell
conda create -n blender_py37 python=3.7
conda activate blender_py37
pip install numpy
pip install bpy
bpy_post_install
python -c "import bpy"
```

For Linux, No-gui env:

```shell
apt-get install subversion
pip install future-fstrings
pip install bpy==2.91a0
bpy_post_install
python -c "import bpy"
```

this throws error, but is still runnable(not 100% sure)

remainders:
TODO

```
rarfile tqdm requests
```

### Addons

Cats: https://github.com/absolute-quantum/cats-blender-plugin

VRM: https://github.com/saturday06/VRM_Addon_for_Blender

For initial addon loading,

```shell
python -c "from datasets.render import Renderer; r = Renderer()"
```

## Dataset

TODO

### Crawling

### Filtering

### Making Train Dataset

## Training

TODO