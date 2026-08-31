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
import requests

# load this directory
sys.path.append(sys.path[0] + '/..')

from agents.remembr_agent import ReMEmbRAgent
from agents.non_agent import NonAgent
from agents.vlm_non_agent import VLMNonAgent

from memory.memory import MemoryItem
from memory.gte_dense_memory import GteDenseMemory
from memory.local_vector_memory import LocalVectorMemory
from memory.qrag_local_memory import QragLocalMemory
from memory.text_memory import TextMemory
from memory.video_memory import VideoMemory, ImageMemoryItem

from tools.tools import format_docs


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return ast.literal_eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
class OllamaTextJudge:
    """Deterministic semantic judge for NaVQA free-form text answers."""

    def __init__(self, model, host='127.0.0.1:11434', num_predict=96, max_retries=3):
        self.model = model
        self.host = host if '://' in host else f'http://{host}'
        self.host = self.host.rstrip('/')
        self.num_predict = num_predict
        self.max_retries = max_retries

    @staticmethod
    def _parse_response(content):
        content = content.strip()
        if content.startswith('```'):
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.I)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def judge(self, question, reference, candidate):
        prompt = f"""You are grading a robot navigation question-answering benchmark.
Decide whether the candidate answer is semantically correct according to the reference answer.
Accept concise paraphrases, harmless extra detail, and equivalent names. Reject contradictions,
wrong attributes, guesses that do not contain the reference fact, and non-answers.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Return exactly one JSON object with a boolean key \"correct\" and a short string key
\"rationale\". Do not use markdown.
/no_think"""
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'think': False,
            'format': 'json',
            'keep_alive': '30m',
            'options': {
                'temperature': 0,
                'seed': 0,
                'num_predict': self.num_predict,
            },
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f'{self.host}/api/generate',
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                raw = response.json().get('response', '')
                result = self._parse_response(raw)
                correct = result.get('correct')
                if not isinstance(correct, bool):
                    raise ValueError(f'Judge returned non-boolean correct value: {correct!r}')
                return {
                    'correct': correct,
                    'rationale': str(result.get('rationale', '')).strip(),
                    'model': self.model,
                }
            except Exception as error:
                last_error = error
                print(f'Text judge attempt {attempt}/{self.max_retries} failed: {error}')
                if attempt < self.max_retries:
                    time.sleep(attempt)
        raise RuntimeError(f'Text judge failed after {self.max_retries} attempts') from last_error


def evaluate_output(qa_instance, predicted, text_judge=None):

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
        references = qa_instance['answers']['text']
        candidate = predicted.get('text')
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError('Model did not return a non-empty text answer')
        if text_judge is None:
            raise ValueError('A text judge is required to score descriptive text questions')
        judge_result = text_judge.judge(
            qa_instance['question'], references[0], candidate.strip()
        )
        out_error = {
            'text_iscorrect': int(judge_result['correct']),
            'text_references': references,
            'text_judge': judge_result,
        }

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(
    model, question, qa_instance, max_retries=3, text_judge=None
):

    print(f'Question: {question}')

    parsed = None
    last_error = None
    elapsed = None
    trace_start = len(model.get_retrieval_trace()) if hasattr(model, 'get_retrieval_trace') else 0
    retrieval_attempts = []
    for attempt in range(1, max_retries + 1):
        attempt_trace_start = (
            len(model.get_retrieval_trace())
            if hasattr(model, 'get_retrieval_trace')
            else 0
        )
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            out_error = evaluate_output(qa_instance, parsed, text_judge=text_judge)
            print("Time elapsed", elapsed)

        except Exception as e:
            last_error = e
            attempt_trace = (
                model.get_retrieval_trace()[attempt_trace_start:]
                if hasattr(model, 'get_retrieval_trace')
                else []
            )
            retrieval_attempts.append(
                {
                    'attempt': attempt,
                    'status': 'failed',
                    'error': f'{type(e).__name__}: {e}',
                    'calls': attempt_trace,
                }
            )
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            print(f"Question attempt {attempt}/{max_retries} failed")
            continue

        attempt_trace = (
            model.get_retrieval_trace()[attempt_trace_start:]
            if hasattr(model, 'get_retrieval_trace')
            else []
        )
        retrieval_attempts.append(
            {
                'attempt': attempt,
                'status': 'succeeded',
                'error': None,
                'calls': attempt_trace,
            }
        )

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed
        return_dict['retrieval_trace'] = (
            model.get_retrieval_trace()[trace_start:]
            if hasattr(model, 'get_retrieval_trace')
            else []
        )
        return_dict['retrieval_attempts'] = retrieval_attempts
        return_dict['candidate_pool'] = (
            model.get_candidate_pool_metadata()
            if hasattr(model, 'get_candidate_pool_metadata')
            else {}
        )

        return return_dict

    failure = {
        "response": parsed,
        "error": {},
        "elapsed": elapsed,
        "evaluation_failure": (
            f"Question failed after {max_retries} attempts: {last_error}"
        ),
        "retrieval_trace": (
            model.get_retrieval_trace()[trace_start:]
            if hasattr(model, 'get_retrieval_trace')
            else []
        ),
        "retrieval_attempts": retrieval_attempts,
        "candidate_pool": (
            model.get_candidate_pool_metadata()
            if hasattr(model, 'get_candidate_pool_metadata')
            else {}
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
            if args.text_retriever != 'dense':
                raise ValueError(
                    '--text_retriever gte_dense/qrag_static/qrag currently requires '
                    '--memory_backend local'
                )
            from memory.milvus_memory import MilvusMemory
            memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time)
        elif args.text_retriever in {'qrag_static', 'qrag'}:
            memory = QragLocalMemory(
                model_name=args.gte_model,
                source_checkpoint=args.qrag_checkpoint,
                inference_checkpoint=args.qrag_inference_checkpoint,
                time_offset=start_time,
                evidence_budget=args.qrag_evidence_budget,
                state_format=args.qrag_state_format,
                retrieval_mode=(
                    'static'
                    if args.text_retriever == 'qrag_static'
                    else 'sequential'
                ),
                episode_mode=args.qrag_episode_mode,
                question_evidence_budget=args.qrag_question_evidence_budget,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
            )
        elif args.text_retriever == 'gte_dense':
            memory = GteDenseMemory(
                model_name=args.gte_model,
                time_offset=start_time,
                text_episode_mode=args.text_episode_mode,
                question_text_evidence_budget=args.question_text_evidence_budget,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
            )
        else:
            memory = LocalVectorMemory(
                embedding_model=args.embedding_model,
                time_offset=start_time,
                text_episode_mode=args.text_episode_mode,
                question_text_evidence_budget=args.question_text_evidence_budget,
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

    learned_caption_embeddings = None
    if isinstance(memory, (GteDenseMemory, QragLocalMemory)):
        learned_caption_embeddings = memory.caption_embeddings(
            captions_path,
            out,
            cache_dir=args.embedding_cache_dir,
        )

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
            # Dynamic attributes keep the Milvus schema backward compatible
            # while giving LocalVectorMemory stable, full-sequence identifiers.
            entity.entry_id = i
            entity.source_file_start = item.get('file_start')
            entity.source_file_end = item.get('file_end')

        if use_milvus:
            text_embedding = (
                learned_caption_embeddings[i]
                if learned_caption_embeddings is not None
                else item['text_embedding']
            )
            memory.insert(entity, text_embedding=text_embedding)
        else:
            memory.insert(entity)

    if hasattr(memory, 'set_candidate_pool_metadata'):
        memory.set_candidate_pool_metadata(
            {
                'sequence_id': args.sequence_id,
                'caption_file': args.caption_file,
                'start_index': int(start_idx),
                'end_index': int(end_idx),
                'count': int(end_idx - start_idx + 1),
                'question_start_time': float(start_time),
                'question_end_time': float(end_time),
                'text_retriever': args.text_retriever,
                'text_episode_mode': (
                    args.qrag_episode_mode
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else args.text_episode_mode
                ),
                'question_text_evidence_budget': (
                    (
                        args.qrag_question_evidence_budget
                        if args.qrag_question_evidence_budget is not None
                        else args.qrag_evidence_budget
                    )
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else args.question_text_evidence_budget
                ),
                'embedding_model': (
                    args.gte_model
                    if args.text_retriever in {'gte_dense', 'qrag_static', 'qrag'}
                    else args.embedding_model
                ),
            }
        )

    if isinstance(memory, QragLocalMemory):
        memory.set_qrag_context(qa_instance['question'])

    if use_optimal_context:
        # then replace the full memory with the optimal context
        memory = TextMemory()
        memory.insert(qa_instance['context'])


    return memory, outputs

def main(args):

    if args.qrag_evidence_budget is None:
        args.qrag_evidence_budget = 1 if args.text_retriever == 'qrag' else 5

    # Questions contain human-readable clock times and the time-search tool
    # parses clock strings back into Unix timestamps.  Pin both operations to
    # the timezone used by the NaVQA annotations instead of inheriting the
    # compute container's timezone.
    zoneinfo_path = os.path.join('/usr/share/zoneinfo', args.timezone)
    if args.timezone.startswith('/') or '..' in args.timezone.split('/'):
        raise ValueError(f"Invalid timezone name: {args.timezone}")
    if not os.path.isfile(zoneinfo_path):
        raise ValueError(
            f"Timezone {args.timezone!r} is not installed at {zoneinfo_path}"
        )
    os.environ['TZ'] = args.timezone
    time.tzset()
    print(f"Using NaVQA annotation/display timezone: {args.timezone}")

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
            max_retrieval_rounds=args.max_retrieval_rounds,
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

    text_judge = None
    if args.text_judge_model:
        text_judge = OllamaTextJudge(
            model=args.text_judge_model,
            host=args.text_judge_host,
            num_predict=args.text_judge_num_predict,
            max_retries=args.text_judge_max_retries,
        )
        print(
            f'Free-form text answers will be judged by {args.text_judge_model} '
            f'at {args.text_judge_host}'
        )


    questions_root = args.questions_dir or os.path.join(args.data_dir, 'questions')
    data_path = os.path.join(questions_root, str(args.sequence_id), args.qa_file+'.json')

    question_payload = json.load(open(data_path, 'r'))
    question_timezone = question_payload.get('metadata', {}).get(
        'timestamp_timezone'
    )
    if question_timezone and question_timezone != args.timezone:
        raise ValueError(
            f"Question file timezone {question_timezone!r} does not match "
            f"evaluation timezone {args.timezone!r}"
        )
    data = question_payload['data']
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

    running_text_successes = 0
    num_text = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []

    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
    os.makedirs(out_path, exist_ok=True)
    retriever_tag = (
        '' if args.text_retriever == 'dense' else f'__{args.text_retriever}'
    )
    name = args.model+'__'+args.caption_file+retriever_tag+args.postfix
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
            elif qa_instance['type'] == 'text' and 'text_iscorrect' in error_dict:
                num_text += 1
                running_text_successes += error_dict['text_iscorrect']
        print(f"Resuming {output_file} at question {len(responses)}/{len(data)}")

        if len(responses) == len(data) and not previous_output.get('in_progress', True):
            print("Evaluation is already complete")
            return

    def save_snapshot(in_progress):
        metrics = {
            "questions_total": len(data),
            "questions_completed": len(responses),
            "questions_scored": num_binary + num_text + num_position + num_time + num_duration,
            "questions_failed": sum('evaluation_failure' in item for item in responses),
            "text_questions_skipped": sum(
                1
                for qa_instance, response in zip(data, responses)
                if qa_instance['type'] == 'text'
                and not response.get('error', {}).get('text_iscorrect') in (0, 1)
                and 'evaluation_failure' not in response
            ),
            "binary_count": num_binary,
            "binary_accuracy": running_successes / num_binary if num_binary else None,
            "text_count": num_text,
            "text_accuracy": running_text_successes / num_text if num_text else None,
            "descriptive_count": num_binary + num_text,
            "descriptive_accuracy": (
                (running_successes + running_text_successes) / (num_binary + num_text)
                if num_binary + num_text else None
            ),
            "position_count": num_position,
            "position_mean_l2_error": running_pos_error / num_position if num_position else None,
            "time_count": num_time,
            "time_mean_absolute_error": running_time_error / num_time if num_time else None,
            "duration_count": num_duration,
            "duration_mean_absolute_error": running_duration_error / num_duration if num_duration else None,
        }
        out_json = {
            "version": 0.5,
            "in_progress": in_progress,
            "config": {
                "timezone": args.timezone,
                "questions_file": data_path,
                "answer_model": args.model,
                "num_predict": args.num_predict,
                "disable_thinking": args.disable_thinking,
                "text_judge_model": args.text_judge_model,
                "text_judge_host": args.text_judge_host,
                "text_judge_num_predict": args.text_judge_num_predict,
                "retrieval_trace": args.memory_backend == 'local',
                "controller_retrieval_mode": "interleaved_single_call",
                "max_retrieval_rounds": args.max_retrieval_rounds,
                "text_retriever": args.text_retriever,
                "embedding_model": (
                    args.gte_model
                    if args.text_retriever in {'gte_dense', 'qrag_static', 'qrag'}
                    else args.embedding_model
                ),
                "embedding_device": args.embedding_device,
                "embedding_cache_dir": args.embedding_cache_dir,
                "text_episode_mode": (
                    args.qrag_episode_mode
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else args.text_episode_mode
                ),
                "question_text_evidence_budget": (
                    (
                        args.qrag_question_evidence_budget
                        if args.qrag_question_evidence_budget is not None
                        else args.qrag_evidence_budget
                    )
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else args.question_text_evidence_budget
                ),
                "qrag_checkpoint": (
                    args.qrag_checkpoint
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
                "qrag_inference_checkpoint": (
                    args.qrag_inference_checkpoint
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
                "qrag_state_format": (
                    args.qrag_state_format
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
                "qrag_evidence_budget": (
                    args.qrag_evidence_budget
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
                "qrag_episode_mode": (
                    args.qrag_episode_mode
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
                "qrag_question_evidence_budget": (
                    (
                        args.qrag_question_evidence_budget
                        if args.qrag_question_evidence_budget is not None
                        else args.qrag_evidence_budget
                    )
                    if args.text_retriever in {'qrag_static', 'qrag'}
                    else None
                ),
            },
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

        if qa_instance['type'] == 'text' and text_judge is None:
            raise ValueError(
                'This evaluation includes text questions; pass --text_judge_model '
                'to score them instead of silently skipping them.'
            )

        memory, instance_captions = load_memory(args, data[i], use_milvus=use_milvus, use_optimal_context=use_optimal_context, ip_address=args.db_ip)
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))

        # model.update_for_instance(captions=instance_captions, ref_time=start_time)
        agent.set_memory(memory)


        out_dict = answer_squad_question(
            agent,
            question,
            qa_instance,
            max_retries=args.max_retries,
            text_judge=text_judge,
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
        elif qa_instance['type'] == 'text':
            if 'text_iscorrect' in error_dict:
                num_text += 1
                running_text_successes += error_dict['text_iscorrect']

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes/num_binary if num_binary else None)
        print("Running Text QA accuracy", running_text_successes/num_text if num_text else None)
        print(
            "Running Descriptive QA accuracy",
            (running_successes + running_text_successes) / (num_binary + num_text)
            if num_binary + num_text else None,
        )
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
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/Los_Angeles",
        help="IANA timezone used by NaVQA's human-readable timestamps",
    )

    parser.add_argument("--out_dir", type=str, default="./out/")

    parser.add_argument("--postfix", type=str, default='_0')


    # all model args
    # parser.add_argument("--use_gt_context", type=bool, default=False)


    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_ctx", type=int, default=8192*8)
    parser.add_argument("--num_predict", type=int, default=2048)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument(
        "--text_judge_model",
        type=str,
        default=None,
        help="Ollama model used as a semantic judge for free-form text questions",
    )
    parser.add_argument(
        "--text_judge_host",
        type=str,
        default=os.environ.get('OLLAMA_HOST', '127.0.0.1:11434'),
    )
    parser.add_argument("--text_judge_num_predict", type=int, default=96)
    parser.add_argument("--text_judge_max_retries", type=int, default=3)

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')
    parser.add_argument("--memory_backend", choices=['local', 'milvus'], default='local')
    parser.add_argument(
        "--text_retriever",
        choices=['dense', 'gte_dense', 'qrag_static', 'qrag'],
        default='dense',
        help=(
            "Text ranker ablation. 'dense' preserves the existing mxbai/L2 "
            "path; 'gte_dense' uses unfine-tuned GTE-base/cosine retrieval; "
            "'qrag_static' applies Q-RAG weights once for static top-k; "
            "'qrag' uses fixed-step sequential zero-shot Q-RAG retrieval."
        ),
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default='mixedbread-ai/mxbai-embed-large-v1',
    )
    parser.add_argument(
        "--gte_model",
        type=str,
        default='Alibaba-NLP/gte-multilingual-base',
        help="Hugging Face model ID or local snapshot for the B1 retriever",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default='cpu',
        help="Torch device used by gte_dense, for example cpu or cuda:0",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default=None,
        help="Optional writable directory for validated full-sequence GTE vectors",
    )
    parser.add_argument(
        "--text_episode_mode",
        choices=['per_call', 'question'],
        default='per_call',
        help=(
            "For dense/GTE text retrieval, either rank independently per tool "
            "call or enforce a unique question-level evidence budget"
        ),
    )
    parser.add_argument(
        "--question_text_evidence_budget",
        type=int,
        default=5,
        help="Maximum unique dense/GTE text memories per answer attempt",
    )
    parser.add_argument(
        "--qrag_checkpoint",
        type=str,
        default=None,
        help="Official full Q-RAG checkpoint, used for provenance validation",
    )
    parser.add_argument(
        "--qrag_inference_checkpoint",
        type=str,
        default=None,
        help="Slim state/action checkpoint exported for Q-RAG inference",
    )
    parser.add_argument(
        "--qrag_evidence_budget",
        "--qrag_steps",
        dest="qrag_evidence_budget",
        type=int,
        default=None,
        choices=[1, 3, 5],
        help=(
            "Maximum Q-RAG actions in one text-tool call; defaults to one for "
            "interleaved B3 and five for static B2"
        ),
    )
    parser.add_argument(
        "--qrag_state_format",
        choices=['native', 'controller'],
        default='controller',
        help="Q-RAG state: question+evidence, optionally including tool query",
    )
    parser.add_argument(
        "--qrag_episode_mode",
        choices=['per_call', 'question'],
        default='question',
        help=(
            "Use a legacy fresh Q-RAG episode per tool call or carry selected "
            "evidence and masks across calls in the same answer attempt"
        ),
    )
    parser.add_argument(
        "--qrag_question_evidence_budget",
        type=int,
        default=5,
        help=(
            "Maximum unique text memories per answer attempt in question "
            "episode mode; v3 keeps this at five even when each call returns one"
        ),
    )
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument(
        "--question_indices",
        type=str,
        default=None,
        help="Comma-separated zero-based question indices for a stratified benchmark",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--max_retrieval_rounds",
        type=int,
        default=5,
        help=(
            "Maximum executed retrieval calls per answer attempt; controller "
            "turns are interleaved so every call sees the previous result"
        ),
    )
    parser.add_argument("--resume", action="store_true")


    args = parser.parse_args()
    main(args)
