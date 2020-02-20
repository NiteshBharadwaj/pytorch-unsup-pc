#!/bin/sh

python ../dpc/run/create_data_torch.py \
--split_dir=splits/ \
--inp_dir_renders=renders \
--out_dir=tf_records \
--synth_set=$1 \
--image_size=128 \
--store_camera=True \
--store_voxels=False \
--store_depth=True \
--num_views=5