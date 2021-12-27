from collections import defaultdict
import json
import logging
import math
import os
import pickle
import random

import numpy as np

import bpy
import addon_utils
import mathutils

logging.getLogger("bpy").setLevel(logging.WARNING)


# region filtering
##################

def valid_dirs(dir_root, path_save):
    models = sorted([os.path.join(dir_root, file) for file in os.listdir(dir_root)
                     if os.path.isdir(os.path.join(dir_root, file))])

    pmxs = []
    pmds = []
    vrms = []

    for dir_model in models:
        for root, subdirs, files in os.walk(dir_model):
            for file in files:
                break_flag = False
                if file.endswith('.pmx'):
                    pmxs.append(dir_model)
                    break_flag = True
                    break
                elif file.endswith('.pmd'):
                    pmds.append(dir_model)
                    break_flag = True
                    break
                elif file.endswith('.vrm'):
                    vrms.append(dir_model)
                    break_flag = True
                    break

                if break_flag:
                    break

    print(len(models), len(pmxs), len(pmds), len(vrms))

    valid_list = sorted(pmxs + pmds + vrms)
    valid_list = [os.path.basename(dirname) + '\n' for dirname in valid_list]
    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(valid_list)


def extract_shapekeys(dir_model, path_save):
    """find and save shape keys and bones(poses)"""
    # find loadable model
    path_model = ''
    for root, subdirs, files in os.walk(dir_model):
        break_flag = False
        for file in files:
            path_model = os.path.join(root, file)
            if file.endswith('.pmx'):
                break_flag = True
                break
            elif file.endswith('.pmd'):
                break_flag = True
                break
            elif file.endswith('.vrm'):
                break_flag = True
                break

        if break_flag:
            break

    print(path_model)
    # load model
    r = Renderer()
    r.import_model(path_model=path_model)

    data = {key: {} for key in bpy.data.objects.keys()}

    for key, obj in bpy.data.objects.items():
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # get shape keys
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys is not None:
            data[key]['shape_keys'] = obj.data.shape_keys.key_blocks.keys()
            # shape_keys.extend(list(obj.data.shape_keys.key_blocks))

        # get pose(bones)
        if hasattr(obj, 'pose') and hasattr(obj.pose, 'bones') and obj.pose.bones is not None:
            bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='POSE')
            data[key]['bones'] = obj.pose.bones.keys()
        bpy.ops.object.mode_set(mode='OBJECT')

        obj.select_set(False)

    # print(lines)
    with open(path_save, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


##################
# endregion


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
        {'name': 'VRM_Addon_for_Blender-release',
         'url': '',
         'path': './addons/VRM_Addon_for_Blender-release.zip', },
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
    bpy.data.objects['light'].location = mathutils.Vector((4, -4.2, 5))
    # bpy.data.objects['light'].location = mathutils.Vector((-0.02, -3.4, 19.3))
    # bpy.data.lights['light'].radius = 0
    bpy.data.lights['light'].energy = 1000

    bpy.data.objects['camera'].location = mathutils.Vector((0.30241375, -1.050002, 1.37315))
    bpy.data.objects['camera'].rotation_euler = mathutils.Euler((1.5708, 0, 0))

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # engine choosing: https://www.cgdirector.com/best-renderers-render-engines-for-blender/
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = 'CYCLES'


import math


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

        self.current_model = ''

    def import_model(self, path_model: str):
        if self.current_model != path_model:
            print('deleting old model and loading new model')
            clean_blender()
            import_model(path_model)
            self.current_model = path_model

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


# region test

def test_render():
    r = Renderer()
    path = 'samples/ini-T式雪ノ下雪乃ver100/雪ノ下雪乃ver1.00.pmx'
    path = os.path.join(os.getcwd(), path)
    r.import_model(path)
    r.set_output_path(os.path.join(os.getcwd(), 'base.png'))
    r.render()

    # change shapekey
    r.change_shapekey('あ', 1.0)
    r.set_output_path(os.path.join(os.getcwd(), 'a_1.png'))
    r.render()

    # rotate head
    r.poseRig(
        bpy.data.objects['雪ノ下雪乃Ver1.00_arm'],
        [
            ('頭', 'Y', 60)
        ]
    )
    r.set_output_path(os.path.join(os.getcwd(), 'rotate.png'))
    r.render()

    # find head and render
    location = r.find_head_position(bpy.data.objects['雪ノ下雪乃Ver1.00_mesh'])
    bpy.data.objects['camera'].location = mathutils.Vector(location + (0, -1, 0))
    bpy.data.objects['camera'].rotation_euler = mathutils.Euler((math.pi / 2., 0, 0))
    r.set_output_path(os.path.join(os.getcwd(), 'head.png'))
    r.render()


# endregion

if __name__ == '__main__':
    import sys

    # r = Renderer()

    idx_file = 'metadata/nicovideo.txt'
    with open(idx_file, 'r', encoding='utf-8') as f:
        model_idxs = f.readlines()
    model_idx = model_idxs[int(sys.argv[-1])].strip()
    # dir_model = os.path.join(os.getcwd(), 'samples', model_idx)
    dir_root = '/raid/vision/dhchoi/data/3d_models/models'
    dir_model = os.path.join(dir_root, model_idx)
    if os.path.exists(dir_model):
        try:
            print(dir_model)
            path_save = os.path.join('./', 'metadata', 'shape_keys', f'{model_idx}.json')
            os.makedirs(os.path.dirname(path_save), exist_ok=True)
            extract_shapekeys(dir_model, path_save)
        except Exception as e:
            # raise e
            print(e)
            pass

    print('###########finished############')

    # r.set_output_path(os.path.join(os.getcwd(), model_idx + '.png'))
    # obj = [value for key, value in bpy.data.objects.items() if 'mesh' in key][0]  # TODO this should be fixed, not works for all model
    # location = r.find_head_position(obj)
    # bpy.data.objects['camera'].location = mathutils.Vector(location + (0, -1, 0))
    # bpy.data.objects['camera'].rotation_euler = mathutils.Euler((math.pi / 2., 0, 0))
    # bpy.data.objects['light'].location = mathutils.Vector(location + (0, -1.5, 0))
    # bpy.data.lights['light'].energy = 100
    #
    # bpy.context.scene.render.resolution_x = 512
    # bpy.context.scene.render.resolution_y = 512
    # r.render()
