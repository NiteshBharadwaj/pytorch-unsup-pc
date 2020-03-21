#!/bin/sh

python ../dpc/run/create_data_torch_other.py \
--split_dir=splits/ \
--inp_dir_renders=other_data/shapenetcore_viewdata/ \
--out_dir=tf_records \
--synth_set=$1 \
--image_size=128 \
--store_camera=False \
--store_voxels=False \
--store_depth=False \
--num_views=5
