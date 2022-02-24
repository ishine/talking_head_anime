FROM docker.io/pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

COPY . /root/talking_head_anime_2

RUN apt-get update && \
    apt-get install wget xvfb libxrender1 libglib2.0-0 -y && \
    pip install -r /root/talking_head_anime_2/requirements.txt && \
    pip install bpy==2.91a0 && \
    bpy_post_install && \
    pip install pyvirtualdisplay && \
    mkdir /root/talking_head_anime_2/addons && \
    wget https://github.com/michaeldegroot/cats-blender-plugin/archive/master.zip \
    -O /root/talking_head_anime_2/addons/cats-blender-plugin-master.zip


