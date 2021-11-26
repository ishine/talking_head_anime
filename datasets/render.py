import random
import os

import bpy
import addon_utils
import mathutils


def reset_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def clean_blender():
    for dataset in [bpy.data.objects, bpy.data.meshes,
                    bpy.data.materials, bpy.data.textures, bpy.data.images,
                    bpy.data.collections]:
        for key, value in dataset.items():
            if key != 'camera' and key != 'light':
                dataset.remove(value)


def build_camera_light():
    if 'camera' not in bpy.context.scene.objects.keys():
        cam_data = bpy.data.cameras.new('camera')
        cam = bpy.data.objects.new('camera', cam_data)
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam

    if 'light' not in bpy.context.scene.objects.keys():
        light_data = bpy.data.lights.new('light', type='POINT')
        # light_data = bpy.data.lights.new('light', type='SUN')
        light = bpy.data.objects.new('light', light_data)
        bpy.context.collection.objects.link(light)


def set_addons():
    addon_data = [
        {'name': 'VRM_Addon_for_Blender-release',
         'url': '',
         'path': '/root/talking_head_anime/addons/VRM_Addon_for_Blender.zip', },
        {'name': 'mmd_tools',
         'url': '',
         'path': 'addons/mmd_tools-v1.0.1.zip', },
    ]
    for data in addon_data:
        addon_name = data['name']

        # check addon is installed
        installed_module_names = [module.__name__ for module in addon_utils.modules()]
        if not addon_name in installed_module_names:
            addon_path = data['path']
            bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path)

        # check addon is loaded
        is_loaded = addon_utils.check(addon_name)
        if not is_loaded[0] and not is_loaded[1]:
            addon_utils.enable(addon_name)


def render_settings():
    bpy.data.objects['light'].location = mathutils.Vector((4, -4.2, 5))
    bpy.data.lights['light'].energy = 500

    bpy.data.objects['camera'].location = mathutils.Vector((0.30241375, -1.050002, 1.37315))
    bpy.data.objects['camera'].rotation_euler = mathutils.Euler((1.5708, 0, 0))

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # engine choosing: https://www.cgdirector.com/best-renderers-render-engines-for-blender/
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # bpy.context.scene.render.engine = 'CYCLES'


def import_model(path_input: str):
    file_extension = path_input.rsplit('.', 1)[-1]
    if file_extension == 'vrm':
        bpy.ops.import_scene.vrm(filepath=path_input)
    elif file_extension == 'pmx':
        bpy.ops.mmd_tools.import_model(filepath=path_input)
    else:
        raise ValueError(f'file extension {file_extension} not supported')


def set_output_path(path_output):
    bpy.context.scene.render.filepath = path_output


def render():
    bpy.ops.render.render(write_still=True)


class Renderer:
    def __init__(self):
        reset_blender()
        build_camera_light()
        set_addons()
        render_settings()

        self.current_model = ''

    def render(self, path_model, path_output, parameters=None):
        if not self.current_model == path_model:
            print('deleting old model and loading new model')
            clean_blender()
            import_model(path_model)
            self.current_model = path_model

        set_output_path(path_output)

        render()
