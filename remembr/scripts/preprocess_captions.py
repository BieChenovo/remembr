import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys

# load this directory
sys.path.append(sys.path[0] + '/..')
from captioners.vila_captioner import VILACaptioner
from utils.util import get_frames
import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json

import tqdm



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_outputs(output_file, outputs):
    """Atomically checkpoint captions so an interrupted run can be inspected."""
    temporary_file = f"{output_file}.tmp"
    with open(temporary_file, 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder)
    os.replace(temporary_file, output_file)


def run_video_in_segs(args):

    SEQUENCE_ID=args.seq_id

    # load folders
    pkl_files = glob.glob(os.path.join(args.data_path, str(SEQUENCE_ID), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    if not pkl_files:
        raise FileNotFoundError(
            f"No PKL frames found for sequence {SEQUENCE_ID} under {args.data_path}"
        )

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    time_start = times[0]
    for t, file in zip(times, pkl_files):
        if t - time_start > args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    # The original script omitted the final partial segment.
    if current_segment:
        segments.append(current_segment)

    if args.max_segments is not None:
        if args.max_segments < 1:
            raise ValueError("--max-segments must be at least 1")
        segments = segments[:args.max_segments]

    captions_location = args.out_path
    output_file = os.path.join(
        captions_location,
        f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json',
    )
    os.makedirs(captions_location, exist_ok=True)

    outputs = []
    if os.path.exists(output_file) and not args.overwrite:
        if not args.resume:
            print(f"Caption file already exists, skipping: {output_file}")
            return
        with open(output_file, 'r') as f:
            outputs = json.load(f)
        if len(outputs) > len(segments):
            raise ValueError(
                f"Caption file contains {len(outputs)} entries, but only "
                f"{len(segments)} segments were found: {output_file}"
            )
        if len(outputs) == len(segments):
            print(f"Caption file is already complete: {output_file}")
            return
        print(
            f"Resuming sequence {SEQUENCE_ID} at segment {len(outputs)} "
            f"of {len(segments)}"
        )

    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)
    captioner_model = VILACaptioner(args)

    start_index = len(outputs)
    segment_iterator = enumerate(segments[start_index:], start=start_index)
    for i, file_names in tqdm.tqdm(
        segment_iterator,
        total=len(segments),
        initial=start_index,
    ):

        images = []
        # depth = []
        # bboxes = []
        position = []
        rotation = []
        timestamp = []


        for file in file_names:
            with open(file, 'rb') as f:
                data = pkl.load(f)
                data['cam0'] = data['cam0'][:, :, ::-1]

                images.append(PILImage.fromarray(data['cam0'].astype('uint8'), 'RGB'))
                # depth.append(data['stereo'])
                # bboxes.append(data['bbox_3d'])
                position.append(data['position'])
                rotation.append(data['rotation'])
                timestamp.append(data['timestamp'])

        
        position = np.array(position)
        rotation = np.array(rotation)
        rotation = Rotation.from_quat(rotation).as_euler('xyz', degrees=True)
        timestamp = np.array(timestamp)

        # Sample uniformly while retaining both ends of the time segment.
        if len(images) > args.num_video_frames:
            sample_indices = np.linspace(
                0, len(images) - 1, args.num_video_frames, dtype=int
            )
            images = [images[index] for index in sample_indices]

        out_text = captioner_model.caption(images)

        print(out_text)
        filename_start = os.path.basename(file_names[0])
        filename_end = os.path.basename(file_names[-1])


        text_embedding = embedder.embed_query(out_text)

        
        entity = {
            'id': file_names[0],
            'position': position.mean(axis=0),
            'theta': 3.14, # TEMPORARY: We are not using rotation information yet, so just leaving a placeholder
            'time': timestamp.mean(),
            'caption': out_text,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding
        }

        outputs.append(entity)
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(segments):
            save_outputs(output_file, outputs)


if __name__ == "__main__":

    # default_query = "<video>\n You are a wandering around a university campus.\
    #     Please describe in detail what you see in the few seconds of the video. \
    #     Format it to focus on a few key elements. Provide details about each of these in the following format: \
    #     People: \n \
    #     Objects: \n \
    #     Environmental Features: \n \
    #     Activities/Events: \n \
    #     Other details: \n"
    

    default_query = "<video>\n You are a wandering around a university campus.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."

    # default_query = "<video>\n What is the color of the floor?"

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
    parser.add_argument("--captioner_name", type=str, default="Llama-3-VILA1.5-8b")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
    )

    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Only process the first N segments; useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing caption JSON instead of skipping it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an incomplete caption JSON from its last saved segment.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Atomically save after this many segments (default: 10).",
    )

    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be used together")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")


    # add some rules here
    if 'Efficient-Large-Model/VILA1.5-40b' in args.model_path:
        args.conv_mode = 'hermes-2'
    elif 'Efficient-Large-Model/VILA1.5' in args.model_path:
        args.conv_mode = 'vicuna_v1'
    elif 'Llama' in args.model_path:
        args.conv_mode = 'llama_3'
    else:
        # trust the default conv_mode
        args.conv_mode = args.conv_mode

    run_video_in_segs(args)


# python -W ignore caption_segments.py --video-file "/home/aanwar/projects/memory_nav/foundation-nav/tools/isaac/data/102344094/path_vis/output.avi"     --query "<video>\n Please describe what you see in the few seconds of the video." 
