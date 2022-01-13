## Environment

supports linux only. For windows, you may want to remove display of renderer in `datasets/render.py`

use conda env

```shell
conda create -y -n blender_py37 python=3.7
conda activate blender_py37

apt-get install xvfb -y
wget https://github.com/michaeldegroot/cats-blender-plugin/archive/master.zip -O addons/cats-blender-plugin-master.zip
pip install -r requirements.txt
bpy_post_install
```

test with `python -c "import bpy"`

```
rarfile tqdm requests
```

## Dataset

TODO

### Crawling

### Filtering

### Making Train Dataset

## Training

TODO