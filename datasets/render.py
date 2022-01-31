from collections import defaultdict
import logging
import math
import os

import cv2
import numpy as np
import pyvirtualdisplay

import bpy
import addon_utils
import mathutils

from utils.util import suppress_stdout


class Renderer:
    def __init__(self, make_display=False):
        self.clean_blender()
        self.set_addons()
        self.init_camera()
        self.init_light()
        self.set_configs()

        self.make_display = make_display
        if self.make_display:
            self.display = pyvirtualdisplay.Display(size=(1, 1))
            self.display.start()

        self.current_model = ''

    # region settings

    @staticmethod
    def set_configs():
        """sets bpy(blender) configs

        Returns:

        """
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.engine = 'CYCLES'

        # overwrite existing file
        bpy.context.scene.render.use_overwrite = True

        # Transparent background
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        # compression, default=15
        bpy.context.scene.render.image_settings.compression = 15

        # render image size(changes render region)
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        # render samples (closely related to rendering time)
        bpy.context.scene.eevee.taa_render_samples = 8  # default 64

        #
        bpy.context.scene.eevee.use_taa_reprojection = False
        bpy.context.scene.eevee.use_volumetric_lights = False
        bpy.context.scene.eevee.volumetric_samples = 4

        # cycles settings
        # bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 64  # default 256?

    @staticmethod
    @suppress_stdout
    def set_addons():
        logging.disable(logging.CRITICAL)

        addon_data = [
            {'name': 'cats-blender-plugin-master',
             'url': '',
             'path': './addons/cats-blender-plugin-master.zip', },
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

        logging.disable(logging.NOTSET)

    @staticmethod
    def init_camera(camera_name='Camera'):
        if camera_name not in bpy.context.scene.objects.keys():
            cam_data = bpy.data.cameras.new(camera_name)
            cam = bpy.data.objects.new(camera_name, cam_data)
            bpy.context.collection.objects.link(cam)
            bpy.context.scene.camera = cam
        bpy.data.objects[camera_name].location = mathutils.Vector((0, 0, 0))
        bpy.data.objects[camera_name].rotation_euler = mathutils.Euler((math.pi / 2, 0, 0))
        bpy.data.objects[camera_name].data.type = 'ORTHO'
        bpy.data.objects[camera_name].data.ortho_scale = 0.5  # default=6

    @staticmethod
    def init_light(light_name="Light"):
        if light_name not in bpy.context.scene.objects.keys():
            light_data = bpy.data.lights.new(light_name, type='POINT')
            light = bpy.data.objects.new(light_name, light_data)
            bpy.context.collection.objects.link(light)
        bpy.data.objects[light_name].location = mathutils.Vector((0, -10, 0))
        bpy.data.lights[light_name].energy = 100

    # endgreion

    @staticmethod
    @suppress_stdout
    def _import_model(path_input: str):
        logging.disable(logging.CRITICAL)
        file_extension = path_input.rsplit('.', 1)[-1].lower()
        if file_extension == 'pmx' or file_extension == 'pmd':
            bpy.ops.cats_importer.import_any_model(filepath=path_input)
            # bpy.ops.mmd_tools.import_model(filepath=path_input)
        elif file_extension == 'vrm':
            bpy.ops.cats_importer.import_any_model(filepath=path_input)
        else:
            raise ValueError(f'file extension {file_extension} not supported')
        logging.disable(logging.NOTSET)

    def import_model(self, path_model: str):
        path_model = os.path.abspath(path_model)
        if self.current_model != path_model:
            self.clean_blender()
            self.init_camera()
            self.init_light()
            self._import_model(path_input=path_model)

            self.current_model = path_model

    @staticmethod
    @suppress_stdout
    def render():
        bpy.ops.render.render(write_still=True)

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

        # arr_image = arr.reshape(
        #     bpy.context.scene.render.resolution_x,
        #     bpy.context.scene.render.resolution_y, 4)
        #
        # return arr_image
        return arr

    # endgreion

    # region utils

    @staticmethod
    def set_output_path(path_output):
        bpy.context.scene.render.filepath = path_output

    @staticmethod
    @suppress_stdout
    def clean_blender(camera_name="Camera", light_name="Light"):
        logging.disable(logging.CRITICAL)
        for i in range(10):
            for attribute in dir(bpy.data):
                try:
                    bpy_dataset = getattr(bpy.data, attribute)
                    for key, value in bpy_dataset.items():
                        if key != camera_name and key != light_name and key != 'Scripting' and not (
                                attribute == 'texts' and key == 'render.py'):
                            bpy_dataset.remove(value)
                except Exception as e:
                    pass

        # self.current_model = ''
        logging.disable(logging.NOTSET)

    def exit(self):
        if self.make_display:
            if self.display.is_alive():
                self.display.stop()
        # self.purge()
        self.clean_blender()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    @staticmethod
    def change_shapekey(key, value):
        # find first object which has given key
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        for obj in bpy.data.objects:
            obj.select_set(True)
            if hasattr(obj.data, 'shape_keys'):
                if key in obj.data.shape_keys.key_blocks:
                    bpy.context.view_layer.objects.active = obj
                    obj.data.shape_keys.key_blocks[key].value = value
                    break
            obj.select_set(False)

        # change value
        # bpy.context.object.data.shape_keys.key_blocks[key].value = value

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

    @staticmethod
    def find_head_position(obj, head_key='頭'):
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

    @staticmethod
    @suppress_stdout
    def set_camera_position(camera_name="Camera", light_name="Light"):
        for key, obj in bpy.data.objects.items():
            if key == camera_name or key == light_name:
                continue

            # turn off toon, sphere texture - prevent pink render
            if hasattr(obj, 'mmd_root'):
                if hasattr(obj.mmd_root, 'use_toon_texture'):
                    try:
                        # bpy.data.objects[key].mmd_root.use_toon_texture = False
                        obj.mmd_root.use_toon_texture = False
                    except:
                        pass
                if hasattr(obj.mmd_root, 'use_sphere_texture'):
                    try:
                        # bpy.data.objects[key].mmd_root.use_sphere_texture = False
                        obj.mmd_root.use_sphere_texture = False
                    except:
                        pass

            location = None
            try:
                location = Renderer.find_head_position(obj)
            except:
                pass
            if location is not None:
                bpy.data.objects[camera_name].location = mathutils.Vector(location + (0, -0.5, 0))
                bpy.data.objects[camera_name].rotation_euler = mathutils.Euler((math.pi / 2., 0, 0))
                bpy.data.objects[light_name].location = mathutils.Vector(location + (0, -2, 0))

    # endregion


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
    os.makedirs(dir_temp, exist_ok=True)
    r = Renderer()
    r.import_model(model_path)

    # set camera position

    # base image # TODO check if rest pose
    r.set_output_path(os.path.join(dir_temp, 'base.png'))
    r.render()

    # find bpy object with shape_keys
    for key, obj in bpy.data.objects.items():
        # set camera position
        if key == "Camera" or key == "Light":
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

    r.exit()

# endregion
