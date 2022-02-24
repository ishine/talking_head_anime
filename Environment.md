# Environment for Talking Head Anime

Here, we provide 2 methods: `Conda` and `Docker`(recommended).

## Docker

### Build Image

`docker build -f Dockerfile -t talking_head_anime:v0.0.0 .`

### Run Container

use your own input for `YOUR_CONTAINER_NAME` and `YOUR_DATA_DIR`

```docker run -it \ 
--ipc=host --gpus=all \ 
-p 39980:39980 -p 39981:8888 \
-v YOUR_DATA_DIR:/root/talking_head_anime_2/data \ 
--name YOUR_CONTAINR_NAME \
talking_head_anime:v0.0.0
```

## Conda env

Tutorial here is tested with **Ubuntu 18.04** only.

For windows, I'll add some instructions that might be enough, but not sure.

Make sure you are at `conda` environment with `python=3.7`. (Here, named as `blender_py37`)

```shell
conda create -y -n blender_py37 python=3.7
conda activate blender_py37
```

### Training Dependencies

`pip install -r requirements.txt`

#### Pytorch

See https://pytorch.org/ for detailed instruction. Select your preferences and run the install command.

Example:

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

### (Optional) Dataset Dependencies

You can skip this part if you already have appropriate images dataset to train Talking Head Anime.

Appropriate image dataset:

* set of images consist of (model_base, model_shaped, model_shaped_rotated) pairs.
* Detailed description of pairs:
    * model_base: Image of model with rest(base) pose. `base.png`
    * model_shaped: Image of model with its face parts' shape changed. `shape_0.57_0.0_0.5.png`
    * model_shaped_rotated: Image of model of `model_shaped`'s head rotated with XYZ
      axis.`pose_0.57_0.0_0.5_0.41_-0.68_1.png`

<img src="src/images/base.png?v=1" width=256 align="left"/>
<img src="src/images/shape_0.57_0.0_0.5.png" width=256 align="left"/>
<img src="src/images/pose_0.57_0.0_0.5_0.41_-0.68_1.png" width=256 align="left"/>
<br>
<br>

#### Runninng Dataset Turorial

`conda install nb_conda_kernels`

#### Blender related dependencies

Here, we'll install python `blender` module and blender addons needed to open 3d models.

##### Blenderpy (bpy)

For Linux, run

```shell
pip install bpy==2.91a0
bpy_post_install
```

For Windows, run

```
pip install bpy==2.82
bpy_post_install
```

##### Blender Addons

```
mkdir addons
wget https://github.com/michaeldegroot/cats-blender-plugin/archive/master.zip -O addons/cats-blender-plugin-master.zip
```

#### Pyvirtualdisplay

If you are running with non-gui device (or want to run without gui), you must install `pyvirtualdisplay`.

Note that `pyvirtualdisplay` currently does not supports windows.

```shell
apt-get update
apt-get install xvfb -y
apt-get install libxrender1 -y
pip install pyvirtualdisplay
```

