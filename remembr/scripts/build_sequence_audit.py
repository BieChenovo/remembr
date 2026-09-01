#!/usr/bin/env python3
"""Build a question/caption/frame/video audit page for one NaVQA sequence.

The CODa preprocessing output stores one camera frame per pickle.  ReMEmbR
groups those frames into roughly three-second caption segments and uniformly
samples at most six frames per segment.  This script reconstructs that exact
sampling, renders one contact sheet per caption, optionally encodes the full
camera stream as a browser-compatible MP4, and joins the visual data with the
questions, references, predictions, and strict error diagnostics.

Run this with the CODa environment because the pickle payloads contain NumPy
arrays and the environment already provides NumPy and Pillow.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CODA_ROOT = Path(
    "/hpc2hdd/home/yichenwang/datasets/remembr-coda/coda_data"
)
DEFAULT_RESULT_ROOT = Path(
    "/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts/eval_outs"
)
DEFAULT_RESULT_TAG = "full_nothink_256_pst_descriptive_4gpu_resume_v1"
DEFAULT_FFMPEG = Path("/hpc2hdd/home/yichenwang/envs/coda/bin/ffmpeg")
CAPTION_STEM = "captions_VILA1.5-13b_3_secs"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def portable_artifact_path(path: Path) -> str:
    """Return an artifacts-relative path suitable for a published report."""
    parts = path.parts
    if "artifacts" in parts:
        return str(Path(*parts[parts.index("artifacts") :]))
    return path.name


def timestamp_from_path(path: Path) -> float:
    return float(path.stem)


def format_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, second = divmod(seconds, 60)
    hour, minutes = divmod(minutes, 60)
    if hour:
        return f"{hour:d}:{minutes:02d}:{second:02d}"
    return f"{minutes:02d}:{second:02d}"


def local_time(timestamp: float, timezone: ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp, timezone).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def load_frame(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        record = pickle.load(handle)
    frame = record["cam0"]
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unexpected cam0 frame at {path}: {getattr(frame, 'shape', None)}")
    return frame


def rgb_image(bgr_frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(bgr_frame[:, :, ::-1]), "RGB")


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def sample_indices(start: int, end: int, count: int = 6) -> list[int]:
    size = end - start + 1
    if size < 1:
        raise ValueError(f"Invalid caption frame interval: {start}..{end}")
    if size <= count:
        return list(range(start, end + 1))
    return [int(index) for index in np.linspace(start, end, count, dtype=int)]


def render_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    output_path: Path,
    tile_width: int,
) -> None:
    columns = 3
    source_width, source_height = images[0].size
    tile_height = max(1, round(source_height * tile_width / source_width))
    label_height = 34
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "#091321"
    )
    label_font = font(17)
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        resized = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x, y))
        draw.rectangle((x, y + tile_height, x + tile_width, y + tile_height + label_height), fill="#101f34")
        draw.text((x + 9, y + tile_height + 7), label, fill="#e8f1ff", font=label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.jpg")
    sheet.save(temporary, format="JPEG", quality=82, optimize=True, progressive=True)
    os.replace(temporary, output_path)


def build_caption_assets(
    captions: list[dict[str, Any]],
    frame_paths: list[Path],
    frame_by_name: dict[str, int],
    frames_dir: Path,
    sequence_start: float,
    timezone: ZoneInfo,
    tile_width: int,
    render_sheets: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    slim_captions = []
    for caption_index, caption in enumerate(captions):
        start = frame_by_name[caption["file_start"]]
        end = frame_by_name[caption["file_end"]]
        selected_indices = sample_indices(start, end, 6)
        contact_name = f"caption_{caption_index:04d}.jpg"
        contact_path = frames_dir / contact_name
        sample_records = []
        images = []
        labels = []
        for slot, frame_index in enumerate(selected_indices, start=1):
            frame_path = frame_paths[frame_index]
            timestamp = timestamp_from_path(frame_path)
            relative = timestamp - sequence_start
            sample_records.append(
                {
                    "slot": slot,
                    "pkl_file": frame_path.name,
                    "global_frame_index": frame_index,
                    "timestamp": timestamp,
                    "relative_seconds": relative,
                    "relative_clock": format_clock(relative),
                    "absolute_time": local_time(timestamp, timezone),
                }
            )
            if render_sheets and (overwrite or not contact_path.is_file()):
                images.append(rgb_image(load_frame(frame_path)))
                labels.append(f"F{slot}  {format_clock(relative)}  {frame_path.stem[-8:]}")
        if images:
            render_contact_sheet(images, labels, contact_path, tile_width)
        if (caption_index + 1) % 25 == 0 or caption_index + 1 == len(captions):
            print(f"contact sheets: {caption_index + 1}/{len(captions)}", flush=True)

        caption_timestamp = float(caption["time"])
        slim_captions.append(
            {
                "index": caption_index,
                "start_file": caption["file_start"],
                "end_file": caption["file_end"],
                "start_frame_index": start,
                "end_frame_index": end,
                "source_frame_count": end - start + 1,
                "timestamp": caption_timestamp,
                "relative_seconds": caption_timestamp - sequence_start,
                "relative_clock": format_clock(caption_timestamp - sequence_start),
                "absolute_time": local_time(caption_timestamp, timezone),
                "position": caption["position"],
                "caption": caption["caption"],
                "contact_sheet": f"frames/{contact_name}",
                "sample_frames": sample_records,
            }
        )
    return slim_captions


def encode_video(
    frame_paths: list[Path],
    output_path: Path,
    ffmpeg: Path,
    fps: float,
    video_width: int,
    threads: int,
    overwrite: bool,
) -> None:
    if output_path.is_file() and not overwrite:
        print(f"video already exists: {output_path}", flush=True)
        return
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    first = load_frame(frame_paths[0])
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.mp4")
    log_path = output_path.with_suffix(".ffmpeg.log")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        f"scale={video_width}:-2:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-threads",
        str(threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    print("encoding full video with:", " ".join(command), flush=True)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_handle)
        assert process.stdin is not None
        try:
            for index, path in enumerate(frame_paths):
                frame = first if index == 0 else load_frame(path)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                if (index + 1) % 500 == 0 or index + 1 == len(frame_paths):
                    print(f"video frames: {index + 1}/{len(frame_paths)}", flush=True)
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg stopped early; inspect {log_path}") from error
        finally:
            process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with {return_code}; inspect {log_path}")
    os.replace(temporary, output_path)
    print(f"video written: {output_path}", flush=True)


def ground_truth(question: dict[str, Any]) -> Any:
    answers = question.get("answers", {})
    question_type = question["type"]
    if question_type in answers:
        return answers[question_type]
    text = answers.get("text")
    return text[0] if isinstance(text, list) and text else text


def prediction(question: dict[str, Any], response: dict[str, Any]) -> Any:
    value = response.get(question["type"])
    if value is not None:
        return value
    nested = response.get("response")
    return nested.get(question["type"]) if isinstance(nested, dict) else None


def build_questions(
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    sequence_start: float,
    timezone: ZoneInfo,
) -> list[dict[str, Any]]:
    questions_by_id = {question["id"]: question for question in questions}
    if responses and all(response.get("id") in questions_by_id for response in responses):
        pairs = [(questions_by_id[response["id"]], response) for response in responses]
    elif len(questions) == len(responses):
        pairs = list(zip(questions, responses))
    else:
        raise ValueError(
            f"Question/response mismatch without usable response IDs: "
            f"{len(questions)} vs {len(responses)}"
        )
    output = []
    for index, (question, response) in enumerate(pairs, start=1):
        diagnostic = diagnostics[question["id"]]
        start = float(question["start_time"])
        end = float(question["end_time"])
        output.append(
            {
                "index": index,
                "id": question["id"],
                "type": question["type"],
                "category": question.get("category"),
                "length_category": str(question.get("length_category", "unknown")).lower(),
                "length_seconds": question.get("length"),
                "start_time": start,
                "end_time": end,
                "start_relative_seconds": start - sequence_start,
                "end_relative_seconds": end - sequence_start,
                "start_relative_clock": format_clock(start - sequence_start),
                "end_relative_clock": format_clock(end - sequence_start),
                "start_absolute_time": local_time(start, timezone),
                "end_absolute_time": local_time(end, timezone),
                "file_info": question.get("file_info"),
                "question": question["question"],
                "ground_truth": ground_truth(question),
                "all_reference_answers": question.get("answers"),
                "model_prediction": prediction(question, response),
                "model_text": response.get("text")
                or ((response.get("response") or {}).get("text") if isinstance(response.get("response"), dict) else None),
                "response_type": response.get("type")
                or ((response.get("response") or {}).get("type") if isinstance(response.get("response"), dict) else None),
                "raw_response": response,
                "retrieval_trace_recorded": "retrieval_trace" in response,
                "retrieval_trace": response.get("retrieval_trace", []),
                "retrieval_attempts": response.get("retrieval_attempts", []),
                "saved_candidate_pool": response.get("candidate_pool", {}),
                "official_correct": diagnostic["official_correct"],
                "output_status": diagnostic["output_status"],
                "observable_reason": diagnostic["observable_reason"],
                "observable_reason_label": diagnostic["observable_reason_label"],
                "metric_error": diagnostic["metric_error"],
                "threshold": diagnostic["threshold"],
                "unit": diagnostic["unit"],
                "elapsed_seconds": diagnostic["elapsed_seconds"],
                "candidate_start_index": diagnostic["candidate_start_index"],
                "candidate_end_index": diagnostic["candidate_end_index"],
                "candidate_count": diagnostic["candidate_count"],
                "reference_entry_ids": diagnostic["reference_entry_ids"],
                "reference_all_in_candidate_pool": diagnostic[
                    "reference_all_in_candidate_pool"
                ],
                "reference_outside_details": diagnostic["reference_outside_details"],
                "reference_context": question["context"],
                "manual_failure_note": diagnostic.get("manual_failure_note"),
                "evaluation_failure": diagnostic.get("evaluation_failure"),
            }
        )
    return output


def escape_script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(data: dict[str, Any]) -> str:
    embedded = escape_script_json(data)
    generated = html.escape(data["generated_at"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NaVQA 序列 {data['sequence_id']} 逐帧错误盘查</title>
<style>
:root{{--bg:#07111f;--panel:#101e31;--panel2:#162842;--line:#2a405d;--text:#edf5ff;--muted:#99abc2;--blue:#5aabff;--green:#39d899;--red:#ff7280;--orange:#ffb45f;--cyan:#45d9de;--purple:#bb8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 5% 0,#173d69 0,transparent 34rem),var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}button,input,select{{font:inherit}}button{{cursor:pointer}}.wrap{{width:min(1600px,calc(100% - 30px));margin:auto;padding:30px 0 70px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:18px}}h1{{font-size:clamp(28px,4vw,48px);margin:2px 0;line-height:1.08}}h2{{font-size:20px;margin:0 0 14px}}h3{{margin:0 0 8px}}.eyebrow{{color:var(--cyan);font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase}}.muted,small{{color:var(--muted)}}.panel{{background:linear-gradient(145deg,rgba(22,40,66,.97),rgba(11,24,41,.97));border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 18px 45px rgba(0,0,0,.18);margin-bottom:16px}}.video-grid{{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(310px,.55fr);gap:18px}}video{{width:100%;max-height:68vh;background:#000;border-radius:12px}}.facts{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.fact,.answer{{padding:12px;background:#0b1728;border:1px solid #243852;border-radius:10px}}.fact span,.answer span{{display:block;color:var(--muted);font-size:11px}}.fact strong{{display:block;font-size:18px;margin-top:3px}}.question-layout{{display:grid;grid-template-columns:270px minmax(0,1fr);gap:18px}}.question-tools{{display:flex;gap:8px;margin-bottom:10px}}.question-tools button,.jump,.scope button{{border:1px solid var(--line);color:var(--text);background:#0a182a;border-radius:8px;padding:8px 10px}}.question-tools button.active,.scope button.active{{background:#17497a;border-color:#4c9de9}}.q-list{{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;max-height:520px;overflow:auto;padding-right:4px}}.q-button{{border:1px solid var(--line);border-radius:8px;background:#0b1728;color:var(--muted);padding:8px 4px}}.q-button.wrong{{border-color:#713947;color:#ffacb4}}.q-button.correct{{border-color:#246348;color:#85e8b8}}.q-button.selected{{outline:2px solid var(--blue);color:#fff}}.badges{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}}.badge{{display:inline-block;border-radius:99px;padding:3px 9px;background:#243956;color:#c9d7e9;font-size:11px;font-weight:750}}.badge.correct{{background:#174632;color:#78e4b0}}.badge.wrong{{background:#53232c;color:#ff9da8}}.prompt{{white-space:pre-wrap;font-size:17px;background:#091522;border-left:4px solid var(--blue);padding:13px 15px;border-radius:8px}}.answers{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}}.answer b{{display:block;margin-top:5px;overflow-wrap:anywhere}}.jump-row{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}}.jump{{color:#a9d4ff}}details{{border-top:1px solid #293e59;padding-top:10px;margin-top:10px}}summary{{cursor:pointer;font-weight:700}}pre{{white-space:pre-wrap;word-break:break-word;color:#cbd9e8;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}.retrievals{{display:grid;gap:10px;margin:14px 0}}.retrieval{{background:#0a1728;border:1px solid #3e3965;border-left:4px solid var(--purple);border-radius:10px;padding:12px}}.retrieval ol{{margin:8px 0 0;padding-left:25px}}.retrieval li.reference-hit{{color:#79e5b1;font-weight:700}}.caption-head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}.scope{{display:flex;gap:6px}}.caption-head input{{flex:1;min-width:230px;background:#091522;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:9px 11px}}.caption-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}}.caption-card{{border:1px solid var(--line);background:#0b1728;border-radius:12px;overflow:hidden}}.caption-card.reference{{border:2px solid var(--green)}}.caption-card.retrieved{{box-shadow:inset 0 0 0 3px var(--purple)}}.caption-card.outside{{box-shadow:inset 0 0 0 2px var(--red)}}.caption-card.reference.retrieved{{box-shadow:inset 0 0 0 3px var(--purple),0 0 16px rgba(187,140,255,.3)}}.caption-card img{{display:block;width:100%;height:auto;background:#050b12}}.caption-body{{padding:13px}}.caption-meta{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}.caption-text{{font-size:14px}}.frame-files{{color:#899db7;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}}.notice{{padding:12px 14px;background:#292117;border-left:4px solid var(--orange);border-radius:8px;margin:10px 0}}#captionCount{{margin-left:auto}}@media(max-width:1050px){{.video-grid,.question-layout{{grid-template-columns:1fr}}.q-list{{grid-template-columns:repeat(10,1fr);max-height:none}}}}@media(max-width:760px){{.caption-grid,.answers,.facts{{grid-template-columns:1fr}}.q-list{{grid-template-columns:repeat(6,1fr)}}header{{display:block}}}}
</style></head><body><main class="wrap">
<header><div><div class="eyebrow">ReMEmbR · NaVQA forensic viewer</div><h1>序列 {data['sequence_id']} 逐帧错误盘查</h1><p class="muted">完整相机流 → 3 秒窗口 → VILA 六帧输入 → caption → question → prediction</p></div><small>生成于 {generated}</small></header>
<section class="panel video-grid"><div><video id="video" controls preload="metadata" src="{data['video']['path']}"></video><div class="jump-row"><button class="jump" onclick="seekTo(0)">回到序列开头</button><button class="jump" id="jumpQuestionStart">跳到问题起点</button><button class="jump" id="jumpQuestionEnd">跳到问题终点</button></div></div><div><h2>视频与 caption 生成事实</h2><div class="facts"><div class="fact"><span>完整相机帧</span><strong>{data['video']['frame_count']}</strong></div><div class="fact"><span>重建帧率</span><strong>{data['video']['fps']:.3f} FPS</strong></div><div class="fact"><span>序列时长</span><strong>{data['video']['duration_clock']}</strong></div><div class="fact"><span>Caption 数</span><strong>{len(data['captions'])}</strong></div></div><div class="notice"><b>帧不是后来随意截的：</b>每个约 3 秒窗口最多有约 31 个原始帧，captioner 用 <code>linspace</code> 均匀抽取 6 帧，并把六张图共同输入 VILA。下面的 contact sheet 就是这 6 帧。</div><p class="muted">完整 MP4 是从所有 CODa <code>cam0</code> PKL 帧重建的 H.264 视频；时间轴 00:00 对应 {html.escape(data['video']['start_absolute_time'])}。</p></div></section>
<section class="panel"><h2>选择问题</h2><div id="traceNotice" class="notice"></div><div class="question-layout"><aside><div class="question-tools"><button id="allQuestions" class="active">全部问题</button><button id="wrongQuestions">仅错误</button></div><div class="q-list" id="questionList"></div></aside><article id="questionDetail"></article></div></section>
<section class="panel"><div class="caption-head"><h2>Caption 与六帧输入</h2><div class="scope"><button data-scope="window" class="active">当前问题候选池</button><button data-scope="retrieved">实际 Retrieve</button><button data-scope="reference">Reference</button><button data-scope="all">全序列</button></div><input id="captionSearch" type="search" placeholder="搜索 caption 文本"><span id="captionCount" class="muted"></span></div><p class="muted">紫色内框 = 实际 top-k 返回给 reader；绿色边框 = 标注 Reference；红色内框 = Reference 落在候选池外。</p><div id="captionGrid" class="caption-grid"></div></section>
</main><script id="auditData" type="application/json">{embedded}</script><script>
const DATA=JSON.parse(document.getElementById('auditData').textContent),video=document.getElementById('video');
const legacyRenderRetrievals=renderRetrievals;
renderRetrievals=q=>{{const enriched=(q.retrieval_trace||[]).map(call=>{{const control=[call.controller_turn_id!==undefined?`Turn ${{call.controller_turn_id}}`:null,call.tool_batch_id?`Batch ${{call.tool_batch_id}} (size=${{call.tool_batch_size}})`:null,call.retrieval_kind||null,call.duplicate_blocked?'DUPLICATE BLOCKED':null,call.duplicate_reprompted?`REPROMPT ${{call.duplicate_replan_count}}/${{call.duplicate_replan_limit}}`:null,call.evidence_state_version!==undefined?`Evidence v${{call.evidence_state_version}}`:null,call.global_selected_entry_ids_after?`Global IDs [${{call.global_selected_entry_ids_after.join(', ')}}]`:null,call.forced_stop_reason?`STOP ${{call.forced_stop_reason}}`:null].filter(Boolean).join(' · ');return {{...call,query:call.raw_query??call.query,retrieval_method:[call.retrieval_method,control].filter(Boolean).join(' · '),score_name:call.retrieval_executed===false?'not executed':call.score_name,returned_context:call.blocked_reason||call.returned_context}}}});return legacyRenderRetrievals({{...q,retrieval_trace:enriched}})}};
let selectedQuestion=DATA.questions[0],questionMode='all',captionScope='window';
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt=v=>typeof v==='object'&&v!==null?JSON.stringify(v):String(v??'—');
const retrievedIds=q=>[...new Set((q.retrieval_trace||[]).flatMap(call=>(call.selected||[]).map(row=>row.entry_id)).filter(id=>id!==null&&id!==undefined))];
const retrievalHits=(q,id)=>(q.retrieval_trace||[]).flatMap(call=>(call.selected||[]).filter(row=>row.entry_id===id).map(row=>({{call:call.call_index,rank:row.rank,score:row.score}})));
function seekTo(seconds){{video.currentTime=Math.max(0,Number(seconds)||0);video.play().catch(()=>{{}});window.scrollTo({{top:0,behavior:'smooth'}})}}
function renderQuestionList(){{const root=document.getElementById('questionList');root.innerHTML='';DATA.questions.filter(q=>questionMode==='all'||!q.official_correct).forEach(q=>{{const b=document.createElement('button');b.className=`q-button ${{q.official_correct?'correct':'wrong'}} ${{q.id===selectedQuestion.id?'selected':''}}`;b.textContent=`Q${{q.index}}`;b.title=`${{q.type}} · ${{q.observable_reason_label}}`;b.onclick=()=>{{selectedQuestion=q;renderAll()}};root.appendChild(b)}})}}
function metricText(q){{if(q.metric_error===null||q.metric_error===undefined)return q.output_status==='invalid'?'无有效结构化值':'—';return `${{Number(q.metric_error).toFixed(2)}} ${{q.unit||''}}（阈值 ${{q.threshold??'—'}} ${{q.unit||''}}）`}}
function renderRetrievals(q){{const calls=q.retrieval_trace||[];if(!calls.length)return q.retrieval_trace_recorded?'<div class="notice"><b>本题检索调用为 0：</b>trace 记录功能已启用，但 agent 没有调用 memory retrieval 工具；reader 在没有检索证据的情况下直接作答。</div>':'<div class="notice"><b>本题没有 retrieval trace：</b>这是旧结果，无法知道实际 top-k，因此不能严格判定是 retriever 还是 reader 出错。</div>';return `<div class="retrievals">${{calls.map(call=>{{const rows=(call.selected||[]).map(row=>{{const isRef=q.reference_entry_ids.includes(row.entry_id);return `<li class="${{isRef?'reference-hit':''}}"><button class="jump" onclick="seekTo(${{DATA.captions[row.entry_id].relative_seconds}})">#${{row.entry_id}}</button> step=${{row.selection_step||row.rank}}, score=${{Number(row.score).toFixed(4)}}${{isRef?' · Reference 命中':''}} — ${{esc(row.caption)}}</li>`}}).join('');const refRanks=q.reference_entry_ids.map(id=>{{const ranked=(call.ranking||[]).find(row=>row.entry_id===id);return `Reference #${{id}} initial rank=${{ranked?ranked.rank:'不在候选池'}}`}}).join('；'),direction=call.lower_is_better===false?'越大越相似':'越小越相似',steps=(call.steps||[]).map(step=>{{const selectedRef=q.reference_entry_ids.includes(step.selected_entry_id),top=(step.ranking||[]).slice(0,5).map(row=>`<li class="${{q.reference_entry_ids.includes(row.entry_id)?'reference-hit':''}}">#${{row.entry_id}} rank=${{row.rank}}, Q=${{Number(row.score).toFixed(4)}}</li>`).join('');return `<details><summary>Q-RAG step ${{step.step}} · 选择 #${{step.selected_entry_id}}${{selectedRef?' · Reference 命中':''}}</summary><div class="muted">state = ${{esc((step.state_components||[]).join(' [SEP] '))}}</div><ol>${{top}}</ol></details>`}}).join('');return `<div class="retrieval"><b>Call ${{call.call_index}} · ${{esc(call.tool)}}${{call.retrieval_method?' · '+esc(call.retrieval_method):''}}</b><div>query = <code>${{esc(fmt(call.query))}}</code> · ${{esc(call.score_name)}}（${{direction}}）</div><div class="muted">${{esc(refRanks)}}</div><ol>${{rows}}</ol>${{steps}}<details><summary>查看实际返回给 reader 的原始 context</summary><pre>${{esc(call.returned_context)}}</pre></details></div>`}}).join('')}}</div>`}}
function renderQuestion(){{const q=selectedQuestion,refs=q.reference_entry_ids.map(i=>DATA.captions[i]);document.getElementById('questionDetail').innerHTML=`<div class="badges"><span class="badge ${{q.official_correct?'correct':'wrong'}}">${{q.official_correct?'正确':'错误'}}</span><span class="badge">Q${{q.index}} · ${{esc(q.type)}}</span><span class="badge">${{esc(q.length_category)}}</span><span class="badge">${{esc(q.observable_reason_label)}}</span></div><h3>${{esc(q.id)}}</h3><div class="prompt">${{esc(q.question)}}</div><div class="answers"><div class="answer"><span>参考答案</span><b>${{esc(fmt(q.ground_truth))}}</b></div><div class="answer"><span>模型结构化答案</span><b>${{esc(fmt(q.model_prediction))}}</b></div><div class="answer"><span>误差/判分</span><b>${{esc(metricText(q))}}</b></div></div><p><b>模型文本：</b>${{esc(q.model_text)}}</p>${{q.manual_failure_note?`<div class="notice"><b>人工失效诊断：</b>${{esc(q.manual_failure_note)}}</div>`:''}}<p class="muted">问题视频窗口：${{q.start_relative_clock}}–${{q.end_relative_clock}} · evaluator 候选池：#${{q.candidate_start_index}}–#${{q.candidate_end_index}}（${{q.candidate_count}} 条）· 实际 retrieve：${{retrievedIds(q).length}} 个唯一 caption</p>${{renderRetrievals(q)}}<div class="jump-row">${{refs.map(c=>`<button class="jump" onclick="seekTo(${{c.relative_seconds}})">跳到 Reference #${{c.index}} · ${{c.relative_clock}}</button>`).join('')}}</div><details><summary>查看 reference context、原始答案和完整模型响应</summary><h4>Reference context</h4><pre>${{esc(q.reference_context)}}</pre><h4>全部参考答案字段</h4><pre>${{esc(JSON.stringify(q.all_reference_answers,null,2))}}</pre><h4>完整模型响应</h4><pre>${{esc(JSON.stringify(q.raw_response,null,2))}}</pre></details>`;document.getElementById('jumpQuestionStart').onclick=()=>seekTo(q.start_relative_seconds);document.getElementById('jumpQuestionEnd').onclick=()=>seekTo(q.end_relative_seconds)}}
function captionVisible(c,q,search){{if(search&&!c.caption.toLowerCase().includes(search))return false;if(captionScope==='all')return true;if(captionScope==='reference')return q.reference_entry_ids.includes(c.index);if(captionScope==='retrieved')return retrievedIds(q).includes(c.index);return c.index>=q.candidate_start_index&&c.index<=q.candidate_end_index}}
function renderCaptions(){{const root=document.getElementById('captionGrid'),q=selectedQuestion,search=document.getElementById('captionSearch').value.trim().toLowerCase(),visible=DATA.captions.filter(c=>captionVisible(c,q,search));document.getElementById('captionCount').textContent=`显示 ${{visible.length}} / ${{DATA.captions.length}} 条`;root.innerHTML=visible.map(c=>{{const ref=q.reference_entry_ids.includes(c.index),inside=c.index>=q.candidate_start_index&&c.index<=q.candidate_end_index,hits=retrievalHits(q,c.index),retrieved=hits.length>0,files=c.sample_frames.map(f=>`F${{f.slot}} ${{f.pkl_file}}`).join('\\n'),hitText=hits.map(h=>`Call ${{h.call}} rank ${{h.rank}} score ${{Number(h.score).toFixed(4)}}`).join(' · ');return `<article class="caption-card ${{ref?'reference':''}} ${{retrieved?'retrieved':''}} ${{ref&&!inside?'outside':''}}"><img loading="lazy" src="${{esc(c.contact_sheet)}}" alt="caption ${{c.index}} sampled frames"><div class="caption-body"><div class="caption-meta"><b>#${{c.index}} · ${{c.relative_clock}}</b><span>${{retrieved?'Retrieve ':''}}${{ref?'· Reference ':''}}${{ref&&!inside?'· 池外':''}}</span></div>${{retrieved?`<p style="color:var(--purple)">${{esc(hitText)}}</p>`:''}}<p class="caption-text">${{esc(c.caption)}}</p><p class="muted">${{esc(c.absolute_time)}} · position=${{esc(JSON.stringify(c.position.map(v=>Number(v).toFixed(2))))}} · 原窗口 ${{c.source_frame_count}} 帧 → VILA ${{c.sample_frames.length}} 帧</p><pre class="frame-files">${{esc(files)}}</pre><button class="jump" onclick="seekTo(${{c.relative_seconds}})">在完整视频中查看这里</button></div></article>`}}).join('')||'<p class="muted">当前筛选没有 caption；旧结果没有 trace 时，“实际 Retrieve”会为空。</p>'}}
function renderAll(){{renderQuestionList();renderQuestion();renderCaptions()}}
document.getElementById('traceNotice').innerHTML=DATA.questions.some(q=>(q.retrieval_trace||[]).length)?'<b>有检索轨迹：</b>紫色条目是工具实际返回给 reader 的 top-k；每次调用同时显示 query、分数方向、Reference 的完整候选排名和原始 context。':'<b>历史日志边界：</b>旧评测没有保存 query、top-k 分数和实际 caption IDs。候选池只表示允许检索的范围，Reference 来自标注；二者都不等于模型实际读到的证据。';document.getElementById('allQuestions').onclick=()=>{{questionMode='all';document.getElementById('allQuestions').classList.add('active');document.getElementById('wrongQuestions').classList.remove('active');renderQuestionList()}};document.getElementById('wrongQuestions').onclick=()=>{{questionMode='wrong';document.getElementById('wrongQuestions').classList.add('active');document.getElementById('allQuestions').classList.remove('active');if(selectedQuestion.official_correct)selectedQuestion=DATA.questions.find(q=>!q.official_correct);renderAll()}};document.querySelectorAll('.scope button').forEach(b=>b.onclick=()=>{{captionScope=b.dataset.scope;document.querySelectorAll('.scope button').forEach(x=>x.classList.toggle('active',x===b));renderCaptions()}});document.getElementById('captionSearch').addEventListener('input',renderCaptions);renderAll();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", type=int, default=0)
    parser.add_argument("--coda-root", type=Path, default=DEFAULT_CODA_ROOT)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("artifacts/eval_reports/b0/v1/analysis/error_analysis.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--result-tag", default=DEFAULT_RESULT_TAG)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--video-width", type=int, default=816)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--video-threads", type=int, default=2)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sequence = args.sequence_id
    args.captions = args.captions or Path(f"artifacts/captions/{sequence}/captions/{CAPTION_STEM}.json")
    args.questions = args.questions or Path(f"artifacts/questions/{sequence}/human_qa.json")
    args.result = args.result or (
        args.result_root
        / str(sequence)
        / "human_qa"
        / f"remembr+qwen3:8b__{CAPTION_STEM}_{args.result_tag}.json"
    )
    args.output_dir = args.output_dir or Path(
        f"artifacts/eval_reports/b0/v1/sequence_{sequence}_media"
    )

    coda_dir = args.coda_root / str(sequence)
    frame_paths = sorted(coda_dir.glob("*.pkl"), key=timestamp_from_path)
    if not frame_paths:
        raise FileNotFoundError(f"No CODa PKLs found: {coda_dir}")
    frame_by_name = {path.name: index for index, path in enumerate(frame_paths)}
    sequence_start = timestamp_from_path(frame_paths[0])
    sequence_end = timestamp_from_path(frame_paths[-1])
    duration = sequence_end - sequence_start
    fps = (len(frame_paths) - 1) / duration
    timezone = ZoneInfo(args.timezone)

    captions = load_json(args.captions)
    questions = load_json(args.questions)["data"]
    result = load_json(args.result)
    responses = result["responses"]
    analysis = load_json(args.analysis)
    diagnostics = {
        row["question_id"]: row
        for row in analysis["rows"]
        if int(row["sequence"]) == sequence
    }
    if len(diagnostics) != len(questions):
        raise ValueError(f"Expected {len(questions)} diagnostics, got {len(diagnostics)}")

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    video_path = output_dir / "media" / f"sequence_{sequence}_full.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    slim_captions = build_caption_assets(
        captions,
        frame_paths,
        frame_by_name,
        frames_dir,
        sequence_start,
        timezone,
        args.tile_width,
        not args.skip_contact_sheets,
        args.overwrite,
    )

    if not args.skip_video:
        encode_video(
            frame_paths,
            video_path,
            args.ffmpeg,
            fps,
            args.video_width,
            args.video_threads,
            args.overwrite,
        )

    audit_questions = build_questions(
        questions, responses, diagnostics, sequence_start, timezone
    )
    data = {
        "version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sequence_id": sequence,
        "source_paths": {
            "coda_dir": f"CODa/{sequence}",
            "captions": portable_artifact_path(args.captions),
            "questions": portable_artifact_path(args.questions),
            "result": portable_artifact_path(args.result),
            "analysis": portable_artifact_path(args.analysis),
        },
        "caption_sampling": {
            "seconds_per_caption": 3,
            "max_frames": 6,
            "method": "uniform linspace including both ends",
        },
        "video": {
            "path": f"media/{video_path.name}",
            "frame_count": len(frame_paths),
            "fps": fps,
            "duration_seconds": duration,
            "duration_clock": format_clock(duration),
            "start_timestamp": sequence_start,
            "end_timestamp": sequence_end,
            "start_absolute_time": local_time(sequence_start, timezone),
            "end_absolute_time": local_time(sequence_end, timezone),
            "reconstruction": "all CODa cam0 PKL frames, H.264, no frame subsampling",
        },
        "captions": slim_captions,
        "questions": audit_questions,
    }
    dump_json(output_dir / f"sequence_{sequence}_audit.json", data)
    (output_dir / "index.html").write_text(build_html(data), encoding="utf-8")
    print(output_dir / "index.html")
    print(output_dir / f"sequence_{sequence}_audit.json")


if __name__ == "__main__":
    main()
