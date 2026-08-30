#!/usr/bin/env python3
"""Precompute validated Q-RAG action embeddings for NaVQA sequences."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from remembr.memory.qrag_local_memory import QragLocalMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-root", type=Path, required=True)
    parser.add_argument("--caption-file", default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--sequences", default="0,3,4,6,16,21,22")
    parser.add_argument("--gte-model", required=True)
    parser.add_argument("--qrag-checkpoint", required=True)
    parser.add_argument("--qrag-inference-checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    memory = QragLocalMemory(
        model_name=args.gte_model,
        source_checkpoint=args.qrag_checkpoint,
        inference_checkpoint=args.qrag_inference_checkpoint,
        time_offset=0,
        evidence_budget=5,
        state_format="native",
        device=args.device,
        batch_size=args.batch_size,
    )
    manifest = []
    for sequence in [
        int(value) for value in args.sequences.replace(",", " ").split()
    ]:
        captions_path = (
            args.captions_root
            / str(sequence)
            / "captions"
            / f"{args.caption_file}.json"
        )
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        embeddings = memory.caption_embeddings(
            str(captions_path),
            captions,
            cache_dir=args.cache_dir,
        )
        record = {
            "sequence_id": sequence,
            "caption_count": len(captions),
            "embedding_shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }
        manifest.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    output = Path(args.cache_dir) / "precompute_manifest.json"
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "source_checkpoint_sha256": memory.source_checkpoint_sha256,
                "encoder_signature": memory.ENCODER_SIGNATURE,
                "sequences": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
