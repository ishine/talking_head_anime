## Environment

supports linux only. For windows, you may want to remove display of renderer in `datasets/render.py`

use conda env

### Blender

```shell
conda create -y -n blender_py37 python=3.7
conda activate blender_py37

apt-get install xvfb -y
apt-get install libxrender1 -y
mkdir addons
wget https://github.com/michaeldegroot/cats-blender-plugin/archive/master.zip -O addons/cats-blender-plugin-master.zip
pip install -r requirements.txt
bpy_post_install
```

test with `python -c "import bpy"`

### Pytorch

install desired pytorch version.

tested with `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html`

## Dataset

TODO

### Crawling

### Filtering

### Making Train Dataset

## Training

TODO