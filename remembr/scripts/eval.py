import ast
import json
import numpy as np

from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from time import strftime, localtime
import numpy as np
import tqdm

import re
import time
import uuid
import sys
import os, sys
import pickle as pkl
from PIL import Image as PILImage
import glob

from dataclasses import asdict
import argparse
import traceback 

# load this directory
sys.path.append(sys.path[0] + '/..')

from agents.remembr_agent import ReMEmbRAgent
from agents.non_agent import NonAgent
from agents.vlm_non_agent import VLMNonAgent

from memory.memory import MemoryItem
from memory.local_vector_memory import LocalVectorMemory
from memory.text_memory import TextMemory
from memory.video_memory import VideoMemory, ImageMemoryItem

from tools.tools import format_docs


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return ast.literal_eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted):

    out_error = {}

    q_type = qa_instance['type']
    if 'position' in q_type:

        answer = np.array(qa_instance['answers']['position'])

        # compute L2 loss between predicted['binary'] and answer
        if predicted.get('position') is None:
            raise ValueError('Model did not return a position')
        if type(predicted['position']) == str:
            predicted['position'] = ast.literal_eval(predicted['position'])
        pred_pos = np.array(predicted['position'])

        dist = float(np.linalg.norm(answer - pred_pos))

        out_error['position_error'] = dist

    elif 'binary' in q_type:

        answer = qa_instance['answers']['text'][1] # we made this assumption in other examples that binary answer is the second one

        binary = predicted.get('binary')
        if not isinstance(binary, str) or binary.lower() not in ('yes', 'no'):
            raise ValueError('Model did not return a valid yes/no answer')
        # get correct/incorrect label
        if binary.lower() == answer.lower():
            correct = 1
        else:
            correct = 0

        out_error['binary_iscorrect'] = correct

    elif 'time' in q_type:

        answer = np.array(qa_instance['answers']['time'])

        # compute L2 loss between predicted['binary'] and answer
        if predicted.get('time') is None:
            raise ValueError('Model did not return a time')
        if type(predicted['time']) == str:
            predicted['time'] = ast.literal_eval(predicted['time'])
        pred_time = np.array(predicted['time'])

        dist = float(abs(answer - pred_time))

        out_error['time_error'] = dist

    elif 'duration' in q_type:

        answer = np.array(qa_instance['answers']['duration'])

        # compute L2 loss between predicted['binary'] and answer
        if predicted.get('duration') is None:
            raise ValueError('Model did not return a duration')
        if type(predicted['duration']) == str:
            predicted['duration'] = ast.literal_eval(predicted['duration'])
        pred_time = np.array(predicted['duration'])

        dist = float(abs(answer - pred_time))

        out_error['duration_error'] = dist

    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error = {'answer': answer}

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(model, question, qa_instance, max_retries=3):

    print(f'Question: {question}')

    parsed = None
    last_error = None
    elapsed = None
    for attempt in range(1, max_retries + 1):
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            out_error = evaluate_output(qa_instance, parsed)
            print("Time elapsed", elapsed)

        except Exception as e:
            last_error = e
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            print(f"Question attempt {attempt}/{max_retries} failed")
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict

    failure = {
        "response": parsed,
        "error": {},
        "elapsed": elapsed,
        "evaluation_failure": (
            f"Question failed after {max_retries} attempts: {last_error}"
        ),
    }
    if isinstance(parsed, dict):
        failure.update(parsed)
    return failure


def load_memory(args, qa_instance, use_milvus=True, use_optimal_context=False, ip_address='127.0.0.1'):
    # Here we load everything needed to load a MilvusDB instance neatly
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']


    if use_milvus:
        if args.memory_backend == 'milvus':
            from memory.milvus_memory import MilvusMemory
            memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time)
        else:
            memory = LocalVectorMemory(
                embedding_model=args.embedding_model,
                time_offset=start_time,
            )
    elif 'vlm' in args.model:
        memory = VideoMemory()
    else:
        memory = TextMemory()

    memory.reset()


    captions_root = args.captions_dir or os.path.join(args.data_dir, 'captions')
    captions_path = os.path.join(captions_root, str(args.sequence_id), 'captions', f'{args.caption_file}.json')

    with open(captions_path, 'r') as f:
        out = json.load(f)


    outputs = []

    # Compute start idx
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))

    # Compute end idx
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))


    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    for i in range(start_idx, end_idx+1):

        item = out[i]
        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'time': item['time'], 
            'caption': item['caption'],
        }

        outputs.append(entity)

        if type(memory) == VideoMemory:

            qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i]['file_start'])
            qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i+1]['file_start'])

            qa_start_idx = pkl_files.index(qa_start_path)
            qa_end_idx = pkl_files.index(qa_end_path)
            idxs = np.linspace(qa_start_idx, qa_end_idx, 6, dtype=int)

            for pkl_idx in idxs:
                # pkl_path = os.path.join(args.coda_dir, str(args.sequence_id), item['file_start'])
                pkl_path = pkl_files[pkl_idx]
                with open(pkl_path, 'rb') as f:
                    pkl_data = pkl.load(f)
                entity['image'] = PILImage.fromarray(pkl_data['cam0'].astype('uint8'), 'RGB')

            entity = ImageMemoryItem.from_dict(entity)
        else:
            entity = MemoryItem.from_dict(entity)

        if use_milvus:
            memory.insert(entity, text_embedding=item['text_embedding'])
        else:
            memory.insert(entity)

    if use_optimal_context:
        # then replace the full memory with the optimal context
        memory = TextMemory()
        memory.insert(qa_instance['context'])


    return memory, outputs

def main(args):

    use_milvus = False
    use_optimal_context = False
    if 'remembr' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = ReMEmbRAgent(
            llm_type=base_llm,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            num_predict=args.num_predict,
            disable_thinking=args.disable_thinking,
        )
        use_milvus = True

    elif 'optimal' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = NonAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_optimal_context = True
    elif 'vlm' in args.model:
        agent = VLMNonAgent(llm_type='gpt-4o')

    else:
        agent = NonAgent(llm_type=args.model, num_ctx=args.num_ctx*4, temperature=args.temperature)


    questions_root = args.questions_dir or os.path.join(args.data_dir, 'questions')
    data_path = os.path.join(questions_root, str(args.sequence_id), args.qa_file+'.json')

    data = json.load(open(data_path, 'r'))
    data = data['data']
    if args.question_indices:
        if args.max_questions is not None:
            raise ValueError('--question_indices and --max_questions are mutually exclusive')
        indices = [int(value.strip()) for value in args.question_indices.split(',')]
        if not indices or any(index < 0 or index >= len(data) for index in indices):
            raise ValueError(
                f'Question indices must be between 0 and {len(data) - 1}: {indices}'
            )
        if len(indices) != len(set(indices)):
            raise ValueError(f'Question indices must be unique: {indices}')
        data = [data[index] for index in indices]
    elif args.max_questions is not None:
        data = data[:args.max_questions]


    running_successes = 0
    num_binary = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []

    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
    os.makedirs(out_path, exist_ok=True)
    name = args.model+'__'+args.caption_file+args.postfix
    output_file = os.path.join(out_path, f'{name}.json')

    if args.resume and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            previous_output = json.load(f)
        previous_responses = previous_output.get('responses', [])
        if len(previous_responses) > len(data):
            raise ValueError(
                f"Cannot resume {output_file}: it contains "
                f"{len(previous_responses)} responses for {len(data)} questions"
            )
        responses = previous_responses
        for qa_instance, previous_response in zip(data, responses):
            error_dict = previous_response.get('error', {}) if previous_response else {}
            if qa_instance['type'] == 'position' and 'position_error' in error_dict:
                num_position += 1
                running_pos_error += error_dict['position_error']
            elif qa_instance['type'] == 'binary' and 'binary_iscorrect' in error_dict:
                num_binary += 1
                running_successes += error_dict['binary_iscorrect']
            elif qa_instance['type'] == 'time' and 'time_error' in error_dict:
                num_time += 1
                running_time_error += error_dict['time_error']
            elif qa_instance['type'] == 'duration' and 'duration_error' in error_dict:
                num_duration += 1
                running_duration_error += error_dict['duration_error']
        print(f"Resuming {output_file} at question {len(responses)}/{len(data)}")

        if len(responses) == len(data) and not previous_output.get('in_progress', True):
            print("Evaluation is already complete")
            return

    def save_snapshot(in_progress):
        metrics = {
            "questions_total": len(data),
            "questions_completed": len(responses),
            "questions_scored": num_binary + num_position + num_time + num_duration,
            "questions_failed": sum('evaluation_failure' in item for item in responses),
            "text_questions_skipped": sum(item.get('type') == 'text' for item in data),
            "binary_count": num_binary,
            "binary_accuracy": running_successes / num_binary if num_binary else None,
            "position_count": num_position,
            "position_mean_l2_error": running_pos_error / num_position if num_position else None,
            "time_count": num_time,
            "time_mean_absolute_error": running_time_error / num_time if num_time else None,
            "duration_count": num_duration,
            "duration_mean_absolute_error": running_duration_error / num_duration if num_duration else None,
        }
        out_json = {
            "version": 0.1,
            "in_progress": in_progress,
            "metrics": metrics,
            "responses": responses,
        }
        temporary_file = output_file + '.tmp'
        with open(temporary_file, 'w') as f:
            json.dump(out_json, f, indent=4)
        os.replace(temporary_file, output_file)

    for i in tqdm.tqdm(
        range(len(responses), len(data)),
        total=len(data),
        initial=len(responses),
    ):

        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        context = qa_instance['context']
        start_time = qa_instance['start_time']
        answers = qa_instance['answers']
        id = qa_instance['id']

        if (qa_instance['type'] == 'text'):
            print("Skipping text questions for now")
            responses.append({}) # this means skipped!
            save_snapshot(in_progress=True)
            continue

        memory, instance_captions = load_memory(args, data[i], use_milvus=use_milvus, use_optimal_context=use_optimal_context, ip_address=args.db_ip)
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))

        # model.update_for_instance(captions=instance_captions, ref_time=start_time)
        agent.set_memory(memory)


        out_dict = answer_squad_question(
            agent, question, qa_instance, max_retries=args.max_retries
        )


        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id


        error_dict = out_dict['error']

        # keep track of how many of each. usually all CSVs are one type only
        if qa_instance['type'] == 'position':
            if 'position_error' in error_dict:
                num_position += 1
                running_pos_error += error_dict['position_error']
        elif qa_instance['type'] == 'binary':
            if 'binary_iscorrect' in error_dict:
                num_binary += 1
                running_successes += error_dict['binary_iscorrect']
        elif qa_instance['type'] == 'time':
            if 'time_error' in error_dict:
                num_time += 1
                running_time_error += error_dict['time_error']
        elif qa_instance['type'] == 'duration':
            if 'duration_error' in error_dict:
                num_duration += 1
                running_duration_error += error_dict['duration_error']

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes/num_binary if num_binary else None)
        print("Running Spatial Error", running_pos_error/num_position if num_position else None)
        print("Running Temporal Error", running_time_error/num_time if num_time else None)
        print("Running Duration Error", running_duration_error/num_duration if num_duration else None)

        print()


        responses.append(out_dict)
        save_snapshot(in_progress=True)

    save_snapshot(in_progress=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--model", type=str, default="remembr+llama3")
    parser.add_argument("--qa_file", type=str, default="human_qa")
    parser.add_argument("--caption_file", type=str, default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--captions_dir", type=str, default=None)
    parser.add_argument("--questions_dir", type=str, default=None)
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")

    parser.add_argument("--out_dir", type=str, default="./out/")

    parser.add_argument("--postfix", type=str, default='_0')


    # all model args
    # parser.add_argument("--use_gt_context", type=bool, default=False)


    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_ctx", type=int, default=8192*8)
    parser.add_argument("--num_predict", type=int, default=2048)
    parser.add_argument("--disable_thinking", action="store_true")

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')
    parser.add_argument("--memory_backend", choices=['local', 'milvus'], default='local')
    parser.add_argument(
        "--embedding_model",
        type=str,
        default='mixedbread-ai/mxbai-embed-large-v1',
    )
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument(
        "--question_indices",
        type=str,
        default=None,
        help="Comma-separated zero-based question indices for a stratified benchmark",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")


    args = parser.parse_args()
    main(args)
