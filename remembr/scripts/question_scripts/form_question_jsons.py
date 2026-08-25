#!/usr/bin/env python3
"""Combine NaVQA annotations with timestamped captions.

This version intentionally uses only the Python standard library so question
files can be prepared on a CPU/login node without the ReMEmbR ML environment.
"""

import argparse
import bisect
import copy
import csv
import datetime
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from time import localtime, strftime


TYPE_COLUMN = "Type \n(binary, position, time, text)"
TIMESTAMP_COLUMN = "Timestamp \nwith answer"
CATEGORY_COLUMN = "Question\nCategory"
DEFAULT_TIMEZONE = "America/Los_Angeles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption_file", default="captions_Llama-3-VILA1.5-8b_3_secs"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--captions-dir",
        type=Path,
        help="Caption root containing <seq_id>/captions/<caption_file>.json",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        help="Output root; defaults to <data-dir>/questions",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=(
            "IANA timezone used by NaVQA's timezone-less HH:MM:SS annotations "
            f"and human-readable timestamps (default: {DEFAULT_TIMEZONE})"
        ),
    )
    return parser.parse_args()


def configure_timezone(timezone_name):
    """Pin localtime()/mktime() to the timezone used during annotation.

    NaVQA stores annotation timestamps as bare HH:MM:SS strings.  Letting
    Python interpret those strings in the host timezone makes the generated
    questions and answers depend on whether preprocessing runs in California,
    UTC, or China.  ``tzset`` is available on the Linux systems supported by
    this preprocessing script and also works on Python 3.8, where ``zoneinfo``
    is not part of the standard library yet.
    """

    zoneinfo_path = Path("/usr/share/zoneinfo") / timezone_name
    if timezone_name.startswith("/") or ".." in Path(timezone_name).parts:
        raise ValueError(f"Invalid timezone name: {timezone_name}")
    if not zoneinfo_path.is_file():
        raise ValueError(
            f"Timezone {timezone_name!r} is not installed at {zoneinfo_path}"
        )
    os.environ["TZ"] = timezone_name
    time.tzset()


def rounded_position(position, digits):
    return [round(float(value), digits) for value in position]


def format_docs(docs):
    output = ""
    for doc in docs:
        timestamp = strftime("%Y-%m-%d %H:%M:%S", localtime(doc["time"]))
        output += (
            f"At time={timestamp}, the robot was at an average position of "
            f"{rounded_position(doc['position'], 3)}."
            f"The robot saw the following: {doc['caption']}\n\n"
        )
    return output


def parse_answer(annotation, context, qa_pair):
    question_type = annotation[TYPE_COLUMN].strip()
    text_answer = annotation["Text answer"].strip()
    parsable_answer = annotation["Parsable answer"].strip()

    if question_type == "binary":
        return {"text": [parsable_answer, parsable_answer]}
    if question_type == "text":
        return {"text": [text_answer]}
    if question_type == "position" and len(context) == 1:
        return {"position": context[0]["position"]}
    if question_type == "time" and len(context) == 1:
        minutes_ago = round((qa_pair["end_time"] - context[0]["time"]) / 60, 2)
        return {"text": [f"{minutes_ago} minutes ago"], "time": minutes_ago}
    if question_type == "duration":
        return {
            "text": [f"{parsable_answer} minutes"],
            "duration": float(parsable_answer),
        }

    print(
        f"Warning: falling back to the parsable answer for {qa_pair['id']} "
        f"({question_type})"
    )
    return {"text": [text_answer], question_type: parsable_answer}


def previous_caption_index(caption_times, timestamp):
    return max(
        0,
        min(
            len(caption_times) - 1,
            bisect.bisect_right(caption_times, timestamp) - 1,
        ),
    )


def main():
    args = parse_args()
    configure_timezone(args.timezone)
    print(f"Using NaVQA annotation/display timezone: {args.timezone}")
    navqa_dir = args.data_dir / "navqa"
    captions_dir = args.captions_dir or args.data_dir / "captions"
    questions_dir = args.questions_dir or args.data_dir / "questions"

    annotations_by_sequence_and_id = defaultdict(list)
    with (navqa_dir / "data.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("Question", "").strip():
                continue
            sequence_id = int(row["Seq ID"])
            annotations_by_sequence_and_id[(sequence_id, row["UUID"])].append(row)

    qa_files = sorted(
        navqa_dir.glob("*/qa_unfilled.json"), key=lambda path: int(path.parent.name)
    )
    total_written = 0
    for qa_path in qa_files:
        sequence_id = int(qa_path.parent.name)
        caption_path = (
            captions_dir
            / str(sequence_id)
            / "captions"
            / f"{args.caption_file}.json"
        )
        if not caption_path.exists():
            print(f"Skipping sequence {sequence_id}: missing {caption_path}")
            continue

        captions = json.loads(caption_path.read_text(encoding="utf-8"))
        caption_times = [float(Path(item["id"]).stem) for item in captions]
        unfilled_questions = json.loads(qa_path.read_text(encoding="utf-8"))["data"]
        filled_questions = []

        for qa_pair in unfilled_questions:
            annotations = annotations_by_sequence_and_id.get(
                (sequence_id, qa_pair["id"]), []
            )
            for annotation in annotations:
                filled = copy.deepcopy(qa_pair)
                context_captions = []
                context_starts = []
                context_ends = []

                date = strftime("%m/%d/%Y", localtime(filled["start_time"]))
                for hms_time in annotation[TIMESTAMP_COLUMN].split(","):
                    full_time = f"{date} {hms_time.strip()}"
                    timestamp = time.mktime(
                        datetime.datetime.strptime(
                            full_time, "%m/%d/%Y %H:%M:%S"
                        ).timetuple()
                    )
                    caption = captions[previous_caption_index(caption_times, timestamp)]
                    context_captions.append(caption)
                    context_starts.append(caption["file_start"])
                    context_ends.append(caption["file_end"])

                context_starts.sort(key=lambda value: float(Path(value).stem))
                context_ends.sort(key=lambda value: float(Path(value).stem))
                filled["file_info"]["context_start_filename"] = context_starts[0]
                filled["file_info"]["context_end_filename"] = context_ends[-1]

                current_caption = captions[
                    previous_caption_index(caption_times, filled["end_time"])
                ]
                start_string = strftime(
                    "%Y-%m-%d %H:%M:%S", localtime(filled["start_time"])
                )
                current_string = strftime(
                    "%Y-%m-%d %H:%M:%S", localtime(filled["end_time"])
                )
                current_position = rounded_position(current_caption["position"], 2)
                filled["question"] = (
                    f"You started moving at {start_string}. The current time is "
                    f"{current_string} and you are located at {current_position}. \n "
                    f"{annotation['Question']}"
                )
                filled["type"] = annotation[TYPE_COLUMN].strip()
                filled["category"] = annotation[CATEGORY_COLUMN].strip()
                filled["context"] = format_docs(context_captions)
                filled["answers"] = parse_answer(
                    annotation, context_captions, filled
                )
                filled_questions.append(filled)

        output_path = questions_dir / str(sequence_id) / "human_qa.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "version": 0.2,
                    "metadata": {
                        "timestamp_timezone": args.timezone,
                        "timestamp_storage": "unix_seconds_utc",
                    },
                    "data": filled_questions,
                },
                indent=4,
            ),
            encoding="utf-8",
        )
        print(
            f"Sequence {sequence_id}: wrote {len(filled_questions)} questions "
            f"to {output_path}"
        )
        total_written += len(filled_questions)

    print(f"Wrote {total_written} questions total")


if __name__ == "__main__":
    main()
