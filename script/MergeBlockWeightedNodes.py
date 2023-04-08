import torch

import os
import sys
import json
import hashlib
import copy
import traceback

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np


import comfy.samplers
import comfy.sd
from comfy.sd import ModelPatcher
import comfy.utils

import comfy.clip_vision

import model_management
import importlib

import folder_paths

import re
import argparse
from tqdm import tqdm

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = "cond_stage_model.transformer.text_model.embeddings.position_ids"

class MergeBlockWeighted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model0": ("MODEL",),
                              "model1": ("MODEL",),
                              "base_alpha":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_00":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_01":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_02":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_03":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_04":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_05":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_06":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_07":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_08":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_09":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_10":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "IN_11":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "M__00":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT00":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT01":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT02":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT03":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT04":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT05":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT06":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT07":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT08":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT09":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT10":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              "OUT11":("FLOAT",{"default": 0, "min": 0, "max": 1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "MergeBlockWeighted"

    CATEGORY = "MergeBlockWeighted"

    def MergeBlockWeighted(self, model0, model1,base_alpha,
            IN_00,IN_01,IN_02,IN_03,IN_04,IN_05,IN_06,IN_07,IN_08,IN_09,IN_10,IN_11,
            M__00,
            OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11
             ):
        weights=[IN_00,IN_01,IN_02,IN_03,IN_04,IN_05,IN_06,IN_07,IN_08,IN_09,IN_10,IN_11,M__00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11]
        model_out=copy.deepcopy(model0)
        theta_0=model_out.model.state_dict()
        theta_1=model1.model.state_dict()
        
        re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
        re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
        re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12
        
        count_target_of_basealpha = 0
        for key in (tqdm(theta_0.keys(), desc="Stage 1/2")):
            current_alpha=base_alpha
            if "model" in key and key in theta_1:

                # check weighted and U-Net or not
                if weights is not None and 'model.diffusion_model.' in key:
                    # check block index
                    weight_index = -1

                    if 'time_embed' in key:
                        weight_index=0
                    elif '.out.' in key:
                        weight_index=NUM_TOTAL_BLOCKS-1
                    else:
                        m = re_inp.search(key)
                        if m:
                            inp_idx = int(m.groups()[0])
                            weight_index = inp_idx
                        else:
                            m = re_mid.search(key)
                            if m:
                                weight_index = NUM_INPUT_BLOCKS
                            else:
                                m = re_out.search(key)
                                if m:
                                    out_idx = int(m.groups()[0])
                                    weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

                    if weight_index>=NUM_TOTAL_BLOCKS:
                        print(f"error. illegal block index: {key}")
                    if weight_index >= 0:
                        current_alpha = weights[weight_index]
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1

            theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]#合成！

        for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
            if "model" in key and key not in theta_0:
                theta_0.update({key:theta_1[key]})#补全（如果有B有比A多的层的话）
        
        model_out.model.load_state_dict(theta_0)

        return (model_out,)
