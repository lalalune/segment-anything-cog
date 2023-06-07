#!/bin/bash

# check if sam_vit_h_4b8939.pth exists
# if not, download it from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

if [ ! -f sam_vit_h_4b8939.pth ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# now call cog push to deploy the model
cog push r8.im/lalalune/segment-anything-direct