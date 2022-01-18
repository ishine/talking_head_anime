## Environment

### NON-GUI environment

supports linux only. For windows, you may want to remove display of renderer in `datasets/render.py`

```shell
conda create -y -n blender_py37 python=3.7
conda activate blender_py37

apt-get install xvfb -y
apt-get install libxrender1 -y
mkdir addons
wget https://github.com/michaeldegroot/cats-blender-plugin/archive/master.zip -O addons/cats-blender-plugin-master.zip
pip install -r requirements.txt
bpy_post_install

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

```shell
ln -s YOUR_DATA_PATH data
ln -s YOUR_BLENDER_PATH blender
ln -s YOUR_LOG_PATH logs
```

### WITH-GUI environment

TODO

## Dataset

TODO

### Crawling

### Filtering

### Making Train Dataset

## Training

`python train_tha.py`

### Configs

TODO

### Custom Dataset