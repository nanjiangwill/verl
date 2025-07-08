# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import base64

import pandas as pd
import datasets

from PIL import Image
import io


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/projects/polyullm/congkai/data/verl/data/DeepEyes-Datasets-47k/data_0.1.2_visual_toolbox_v2.parquet")
    parser.add_argument("--local_dir", default="/home/projects/polyullm/congkai/data/verl/data/0_1_2_visual_toolbok_v2")

    args = parser.parse_args()

    # df = pd.read_parquet(args.data_path)
    # import pdb;pdb.set_trace()
    all_dataset = datasets.load_dataset("parquet", data_files=[args.data_path])["train"]
    
    # Split dataset: 100 samples for test, rest for train
    test_dataset = all_dataset.select(range(100))
    train_dataset = all_dataset.select(range(100, len(all_dataset)))

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": example["data_source"],
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": example["prompt"][1]['content'] + "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> ",
                    },
                ],
                "images": [Image.open(io.BytesIO(image['bytes'])) for image in example["images"]],
                "ability": example['ability'],
                "reward_model": example['reward_model'],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example["reward_model"]["ground_truth"],
                    "question": example["prompt"][1]['content'],
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_zoom_in_tool": {
                            "create_kwargs": {
                                "image": "data:image/jpeg;base64," + base64.b64encode(example["images"][0]['bytes']).decode('utf-8')
                            },
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    train_dataset = train_dataset.cast_column("images", datasets.Sequence(datasets.Image()))

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.cast_column("images", datasets.Sequence(datasets.Image()))

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
