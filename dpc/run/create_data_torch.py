import startup

import sys
import os
import glob
import re
import random

import numpy as np
from scipy.io import loadmat
from imageio import imread

from skimage.transform import resize as im_resize

from util.fs import mkdir_if_missing

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_dir", default = '', help="Directory path containing the input rendered images.")
parser.add_argument("--inp_dir_renders", default = '', help="Directory path containing the input rendered images.")
parser.add_argument("--inp_dir_voxels",  default = '',help="Directory path containing the input voxels.")
parser.add_argument("--out_dir",  default = '',help="Directory path to write the output.")
parser.add_argument("--synth_set",  default = '',help = "")
parser.add_argument("--store_camera",  default = False ,type=bool, help = "")
parser.add_argument("--store_voxels",  default = False,type=bool, help = "")
parser.add_argument("--store_depth",  default = False,type=bool, help = "")
parser.add_argument("--split_path",  default = '',help="")

parser.add_argument("--num_views",  default = 10,type = int , help="Num of viewpoints in the input data.")
parser.add_argument("--image_size",  default = 64,type = int, help="Input images dimension (pixels) - width & height.")
parser.add_argument("--vox_size",  default = 32,type = int, help="Voxel prediction dimension.")


args = parser.parse_args()

# flags = {}

# flags['split_dir'] = '../../data/splits/' #'Directory path containing the input rendered images.'
# flags['inp_dir_renders'] = '' # Directory path containing the input rendered images.
# flags['inp_dir_voxels'] = '' # Directory path containing the input voxels.
# flags['out_dir'] = 'test_dir' # Directory path to write the output.
# flags['synth_set'] = '03001627'
# flags['store_camera'] =  False 
# flags['store_voxels'] = False 
# flags['store_depth'] = False 
# flags['split_path'] = ''
# flags['num_views'] = 10 # 'Num of viewpoints in the input data.'
# flags['image_size'] = 64 # 'Input images dimension (pixels) - width & height.'
# flags['vox_size'] = 32 # 'Voxel prediction dimension.'

# not sure if needed
# flags.DEFINE_boolean('tfrecords_gzip_compressed', False, 'Voxel prediction dimension.')

FLAGS = args
print(FLAGS)

def read_camera(filename):
    cam = loadmat(filename)
    extr = cam["extrinsic"]
    pos = cam["pos"]
    return extr, pos


def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap*(maxVal-minVal)/(pow(2,16)-1) + minVal
    return dMap

def create_record(synth_set, split_name, models):
    im_size = FLAGS.image_size
    num_views = FLAGS.num_views
    num_models = len(models)

    mkdir_if_missing(FLAGS.out_dir)

#     # address to save the TFRecords file
#     train_filename = "{}/{}_{}.tfrecords".format(FLAGS.out_dir, synth_set, split_name)
#     # open the TFRecords file
#     options = tf_record_options(FLAGS)
#     writer = tf.python_io.TFRecordWriter(train_filename, options=options)

    render_dir = os.path.join(FLAGS.inp_dir_renders, synth_set)
    voxel_dir = os.path.join(FLAGS.inp_dir_voxels, synth_set)
    for j, model in enumerate(models):
        print("{}/{}".format(j, num_models))

#         if FLAGS.store_voxels:
#             voxels_file = os.path.join(voxel_dir, "{}.mat".format(model))
#             voxels = loadmat(voxels_file)["Volume"].astype(np.float32)

#             # this needed to be compatible with the
#             # PTN projections
#             voxels = np.transpose(voxels, (1, 0, 2))
#             voxels = np.flip(voxels, axis=1)

        im_dir = os.path.join(render_dir, model)
        images = sorted(glob.glob("{}/render_*.png".format(im_dir)))

        rgbs = np.zeros((num_views, im_size, im_size, 3), dtype=np.float32)
        masks = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)
        cameras = np.zeros((num_views, 4, 4), dtype=np.float32)
        cam_pos = np.zeros((num_views, 3), dtype=np.float32)
        depths = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)
        
        assert(len(images) >= num_views)

        for k in range(num_views):
            im_file = images[k]
            img = imread(im_file)
            rgb = img[:, :, 0:3]
            mask = img[:, :, [3]]
            mask = mask / 255.0
            if True:  # white background
                mask_fg = np.repeat(mask, 3, 2)
                mask_bg = 1.0 - mask_fg
                rgb = rgb * mask_fg + np.ones(rgb.shape)*255.0*mask_bg
            # plt.imshow(rgb.astype(np.uint8))
            # plt.show()
            rgb = rgb / 255.0
            actual_size = rgb.shape[0]
            if im_size != actual_size:
                rgb = im_resize(rgb, (im_size, im_size), order=3)
                mask = im_resize(mask, (im_size, im_size), order=3)
            rgbs[k, :, :, :] = rgb
            masks[k, :, :, :] = mask

            fn = os.path.basename(im_file)
            img_idx = int(re.search(r'\d+', fn).group())

            if FLAGS.store_camera:
                cam_file = "{}/camera_{}.mat".format(im_dir, img_idx)
                cam_extr, pos = read_camera(cam_file)
                cameras[k, :, :] = cam_extr
                cam_pos[k, :] = pos

            if FLAGS.store_depth:
                depth_file = "{}/depth_{}.png".format(im_dir, img_idx)
                depth = loadDepth(depth_file)
                d_max = 10.0
                d_min = 0.0
                depth = (depth - d_min) / d_max
                depth_r = im_resize(depth, (im_size, im_size), order=0)
                depth_r = depth_r * d_max + d_min
                depths[k, :, :] = np.expand_dims(depth_r, -1)

        # Create a feature
        feature = {"image": rgbs,
                   "mask": masks,
                   "name": model}
#         if FLAGS.store_voxels:
#             feature["vox"] = voxels

        if FLAGS.store_camera:
            # feature["extrinsic"] = _dtype_feature(extrinsic)
            feature["extrinsic"] = cameras
            feature["cam_pos"] = cam_pos

        if FLAGS.store_depth:
            feature["depth"] = depths
            
            
        feature_file = "{}/{}_features.p".format(FLAGS.out_dir, model)
        with open(feature_file, 'wb') as f:
            pickle.dump(feature, f)

#         # Create an example protocol buffer
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#         # Serialize to string and write on the file
#         writer.write(example.SerializeToString())

        
#         plt.imshow(np.squeeze(img[:,:,0:3]))
#         plt.show()
#         plt.imshow(np.squeeze(img[:,:,3]).astype(np.float32)/255.0)
#         plt.show()
        

#     writer.close()
#     sys.stdout.flush()


SPLIT_DEF = [("val", 0.05), ("train", 0.95)]


def generate_splits(input_dir):
    files = [f for f in os.listdir(input_dir) if os.path.isdir(f)]
    models = sorted(files)
    random.shuffle(models)
    num_models = len(models)
    models = np.array(models)
    out = {}
    first_idx = 0
    for k, splt in enumerate(SPLIT_DEF):
        fraction = splt[1]
        num_in_split = int(np.floor(fraction * num_models))
        end_idx = first_idx + num_in_split
        if k == len(SPLIT_DEF)-1:
            end_idx = num_models
        models_split = models[first_idx:end_idx]
        out[splt[0]] = models_split
        first_idx = end_idx
    return out


def load_drc_split(base_dir, synth_set):
    filename = os.path.join(base_dir, "{}.file".format(synth_set))
    lines = [line.rstrip('\n') for line in open(filename)]

    k = 3  # first 3 are garbage
    split = {}
    while k < len(lines):
        _,_,name,_,_,num = lines[k:k+6]
        k += 6
        num = int(num)
        split_curr = []
        for i in range(num):
            _, _, _, _, model_name = lines[k:k+5]
            k += 5
            split_curr.append(model_name)
        split[name] = split_curr

    return split


def generate_records(synth_set):
    base_dir = FLAGS.split_dir
    split = load_drc_split(base_dir, synth_set)

    for key, value in split.items():
        create_record(synth_set, key, value)


def read_split(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines




if __name__ == '__main__':
    generate_records(FLAGS.synth_set)
