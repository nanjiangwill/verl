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

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/projects/polyullm/congkai/data/verl/data/DeepEyes-Datasets-47k/data_0.1.2_visual_toolbox_v2.parquet")
    parser.add_argument("--local_dir", default="/home/projects/polyullm/congkai/data/verl/data/0_1_2_visual_toolbok_v2")

    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    # import pdb;pdb.set_trace()
    
    dataset = []

    for idx, row in df.iterrows():
        data = {
                "data_source": row["data_source"],
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant.\n\n"
                            "# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": row["prompt"][1]['content'] + "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> ",
                    },
                ],
                "images": row["images"],
                "ability": row['ability'],
                "reward_model": row['reward_model'],
                "extra_info": {
                    "split": "data_0.1.2_visual_toolbox_v2",
                    "index": idx,
                    "answer": row["reward_model"]["ground_truth"],
                    "question": row["prompt"][1]['content'],
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_zoom_in_tool": {
                            "create_kwargs": {
                                "image": row["images"][0]
                            },
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
        dataset.append(data)

    # import pdb;pdb.set_trace()
    dataset = pd.DataFrame(dataset)
    train_dataset = dataset.sample(frac=0.8, random_state=42)
    test_dataset = dataset.drop(train_dataset.index)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
