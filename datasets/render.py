from collections import defaultdict
import logging
import math
import os
import pickle
import random

import numpy as np

import bpy
import addon_utils
import mathutils
import pyvirtualdisplay

logging.getLogger("bpy").setLevel(logging.WARNING)


def reset_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def clean_blender():
    for i in range(10):
        for attribute in dir(bpy.data):
            try:
                bpy_dataset = getattr(bpy.data, attribute)
                for key, value in bpy_dataset.items():
                    if key != 'camera' and key != 'light' and key != 'Scripting' and not (
                            attribute == 'texts' and key == 'render.py'):
                        bpy_dataset.remove(value)
            except Exception as e:
                pass


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
        # {'name': 'VRM_Addon_for_Blender-release',
        #  'url': '',
        #  'path': './addons/VRM_Addon_for_Blender-release.zip', },
        # {'name': 'mmd_tools',
        #  'url': '',
        #  'path': './addons/mmd_tools-v1.0.1.1.zip', },
        {'name': 'cats-blender-plugin-master',
         'url': '',
         'path': './addons/cats-blender-plugin-master.zip', },
    ]
    for data in addon_data:
        addon_name = data['name']
        print(addon_name)

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
    bpy.data.objects['light'].location = mathutils.Vector((0, -10, 0))
    bpy.data.lights['light'].energy = 100

    bpy.data.objects['camera'].location = mathutils.Vector((0, -5, 0))
    bpy.data.objects['camera'].rotation_euler = mathutils.Euler((1.5708, 0, 0))

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # engine choosing: https://www.cgdirector.com/best-renderers-render-engines-for-blender/
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # bpy.context.scene.render.engine = 'CYCLES'


def import_model(path_input: str):
    file_extension = path_input.rsplit('.', 1)[-1].lower()
    # if file_extension == 'vrm':
    if False:
        bpy.ops.import_scene.vrm(filepath=path_input)
    elif file_extension == 'pmx' or file_extension == 'pmd' or file_extension == 'vrm':
        bpy.ops.cats_importer.import_any_model(filepath=path_input)
        # bpy.ops.mmd_tools.import_model(filepath=path_input)
    else:
        raise ValueError(f'file extension {file_extension} not supported')


class Renderer:
    def __init__(self):
        clean_blender()
        build_camera_light()
        set_addons()
        render_settings()
        self.display = pyvirtualdisplay.Display()
        self.display.start()

        self.current_model = ''

    def import_model(self, path_model: str):
        if self.current_model != path_model:
            print('deleting old model and loading new model')
            clean_blender()
            import_model(path_model)
            self.current_model = path_model

    def clear(self):
        clean_blender()

    def render_complex(self, path_output, parameters=None):
        self.set_output_path(path_output)

        self.render()

    @staticmethod
    def render():
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def set_output_path(path_output):
        bpy.context.scene.render.filepath = path_output

    def render_to_numpy_array(self):
        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        # for n in tree.nodes:
        #     tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = 185, 285

        # create output node
        v = tree.nodes.new('CompositorNodeViewer')
        v.location = 750, 210
        v.use_alpha = False

        # Links
        links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input

        # render
        bpy.ops.render.render()

        # get viewer pixels
        pixels = bpy.data.images['Viewer Node'].pixels
        print(len(pixels))  # size is always width * height * 4 (rgba)

        # copy buffer to numpy array for faster manipulation
        arr = np.array(pixels[:])
        print(arr.shape)

    def change_shapekey(self, key, value):
        # find first object which has given key
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        for obj in bpy.data.objects:
            obj.select_set(True)
            if hasattr(obj.data, 'shape_keys'):
                if key in obj.data.shape_keys.key_blocks:
                    bpy.context.view_layer.objects.active = obj
                    break
            obj.select_set(False)
        print(bpy.context.object)

        # change value
        bpy.context.object.data.shape_keys.key_blocks[key].value = value

    @staticmethod
    def poseRig(ob, poseTable):
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        bpy.context.view_layer.objects.active = ob  # Make the cube the active object
        ob.select_set(True)  # Select the cube
        bpy.ops.object.mode_set(mode='POSE')

        for (bname, axis, angle) in poseTable:
            pbone = ob.pose.bones[bname]
            # Set rotation mode to Euler XYZ, easier to understand
            # than default quaternions
            pbone.rotation_mode = 'XYZ'
            # Documentation bug: Euler.rotate(angle,axis):
            # axis in ['x','y','z'] and not ['X','Y','Z']
            pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
        bpy.ops.object.mode_set(mode='OBJECT')

    def find_head_position(self, obj, head_key='頭'):
        vgs = defaultdict(list)
        for v in obj.data.vertices:
            for g in v.groups:
                vgs[obj.vertex_groups[g.group].name].append(v.index)

        ids_ = np.asarray(vgs[head_key])
        v_ = [v.co for v in obj.data.vertices]
        v_ = np.asarray(v_)
        vs_ = v_[ids_]
        location = np.mean(vs_, axis=0)

        return location

    def exit(self):
        self.display.stop()
        self.clear()
        reset_blender()


# region test

def test_render(model_path: str, dir_temp: str = './result_temp'):
    """

    Args:
        model_path: path to model. recommended to give as absolute path
        dir_temp: temporary dir to save rendered images. recommended to give as absolute path

    Returns:

    """
    model_path = os.path.abspath(model_path)
    dir_temp = os.path.abspath(dir_temp)
    print(model_path)
    print(dir_temp)
    os.makedirs(dir_temp, exist_ok=True)
    r = Renderer()
    r.import_model(model_path)
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256

    # set camera position
    for key, obj in bpy.data.objects.items():
        if key == 'camera' or key == 'light':
            continue
        print(key, obj)

        # turn off toon, sphere texture - prevent pink render
        bpy.data.objects[key].mmd_root.use_toon_texture = False
        bpy.data.objects[key].mmd_root.use_sphere_texture = False

        try:
            location = r.find_head_position(obj)
            bpy.data.objects['camera'].location = mathutils.Vector(location + (0, -1, 0))
            bpy.data.objects['camera'].rotation_euler = mathutils.Euler((math.pi / 2., 0, 0))
            bpy.data.objects['light'].location = mathutils.Vector(location + (0, -2, 0))
            bpy.data.lights['light'].energy = 100
        except:
            pass

    # base image # TODO check if rest pose
    r.set_output_path(os.path.join(dir_temp, 'base.png'))
    r.render()

    # find bpy object with shape_keys
    for key, obj in bpy.data.objects.items():
        # set camera position
        if key == 'camera' or key == 'light':
            continue
        print(key, obj)
        # direction = -y -> +y

        # select single object
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # object with shape_keys
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys is not None:
            for shape_key in ['あ', 'ウィンク', 'ウィンク右']:
                try:
                    r.change_shapekey(shape_key, 1.0)
                    r.set_output_path(os.path.join(dir_temp, f'{key}.{shape_key}.png'))
                    r.render()
                    r.change_shapekey(shape_key, 0.0)
                except Exception as e:
                    print(e)
                    pass

        # object with poses
        print(obj)
        if hasattr(obj, 'pose') and hasattr(obj.pose, 'bones') and obj.pose.bones is not None:
            # rotate head
            r.poseRig(
                obj,
                [
                    ('頭', 'Y', 60)
                ]
            )
            r.set_output_path(os.path.join(dir_temp, f'{key}.Y60.png'))
            r.render()

            r.poseRig(
                obj,
                [
                    ('頭', 'Y', -60)
                ]
            )
        obj.select_set(False)


# endregion

if __name__ == '__main__':
    # test_render('samples/3d.nicovideo__10003__こんにゃく式戌亥とこver1.0/こんにゃく式戌亥とこver1.0/戌亥とこ.pmx')
    from tqdm import tqdm
    from datasets.utils.filter import find_model_in_dir
    import sys

    # dir_root = '/raid/vision/dhchoi/data/3d_models/models'
    # dir_save_root = '/raid/vision/dhchoi/data/3d_models/test_images/'
    dir_root = '/DATA/vision/home/dhchoi/data/3d_models/models'
    dir_save_root = './temp_result'
    # for idx, dirname in enumerate(tqdm(os.listdir(dir_root))):
    idx = int(sys.argv[-1])
    if True:
        dirname = os.listdir(dir_root)[idx]
        dir_model = os.path.join(dir_root, dirname)
        result, path_model = find_model_in_dir(dir_model)
        if result:
            dir_save = os.path.join(dir_save_root, dirname)
            os.makedirs(dir_save, exist_ok=True)
            try:
                test_render(path_model, dir_save)
            except Exception as e:
                print(e)
