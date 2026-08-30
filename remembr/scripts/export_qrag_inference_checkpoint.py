#!/usr/bin/env python3
"""Export the two encoders needed for greedy Q-RAG retrieval.

The official training checkpoint also contains target networks, optimizer
moments, and scheduler state.  Those make it roughly 11 GB and are not used by
the evaluation actor.  This script preserves the exact policy state encoder and
critic action encoder used by ``PQN.select_action(..., evaluate=True)``.
"""

import argparse
import hashlib
import os
import tempfile


EXPECTED_SOURCE_SHA256 = (
    "ff2ab1db095fe05f0f854672a224e9dff6ff0a9a8ecda5cf35cdf88a94d37c56"
)
EXPECTED_QRAG_COMMIT = "42358d78ac491843763b90677f07237471c97086"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_prefix(state_dict, prefix):
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def validate_encoder_state(name, state_dict):
    if len(state_dict) != 140:
        raise ValueError(f"Expected 140 {name} tensors, got {len(state_dict)}")
    expected_shapes = {
        "cls_token": (1,),
        "sep_token": (1,),
        "head.weight": (384, 768),
        "head.bias": (384,),
        "model.embeddings.word_embeddings.weight": (250048, 768),
        "model.encoder.layer.11.mlp_ln.bias": (768,),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in state_dict:
            raise ValueError(f"Missing {name} tensor {key}")
        shape = tuple(state_dict[key].shape)
        if shape != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape for {key}: {shape}, "
                f"expected {expected_shape}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    args = parser.parse_args()

    import torch

    source = os.path.realpath(args.source)
    output = os.path.realpath(args.output)
    source_sha256 = sha256_file(source)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError(
            f"Q-RAG checkpoint SHA-256 mismatch: {source_sha256}; "
            f"expected {args.expected_source_sha256}"
        )

    # The source is the pinned official Q-RAG artifact whose digest was checked
    # immediately above. OmegaConf containers in optimizer state require full
    # unpickling, even though only the two state_dict entries are retained.
    checkpoint = torch.load(
        source,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state_encoder = strip_prefix(checkpoint["policy"], "state_embed.")
    action_encoder = strip_prefix(
        checkpoint["critic"],
        "action_embed.model.",
    )
    validate_encoder_state("state encoder", state_encoder)
    validate_encoder_state("action encoder", action_encoder)

    payload = {
        "version": 1,
        "state_encoder": state_encoder,
        "action_encoder": action_encoder,
        "metadata": {
            "source_checkpoint": source,
            "source_checkpoint_sha256": source_sha256,
            "qrag_code_commit": EXPECTED_QRAG_COMMIT,
            "huggingface_model_id": "Q-RAG/qrag-ft-gte-on-hotpotqa_musique",
            "base_encoder": "Alibaba-NLP/gte-multilingual-base",
            "training_datasets": ["HotpotQA", "Musique"],
            "training_max_steps": 6,
            "action_max_length": 256,
            "state_component_max_length": 256,
            "positions_processor": "none",
            "state_encoder_source": "checkpoint['policy'].state_embed",
            "action_encoder_source": "checkpoint['critic'].action_embed.model",
            "pooling": "attention_masked_mean_div_10",
            "embedding_dimension": 768,
            "score": "state_action_dot_product",
            "head_is_unused_by_upstream_forward": True,
        },
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output),
        prefix=".qrag-inference-",
        suffix=".pt",
        delete=False,
    )
    temporary_path = temporary.name
    temporary.close()
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    output_sha256 = sha256_file(output)
    with open(f"{output}.sha256", "w") as stream:
        stream.write(f"{output_sha256}  {output}\n")
    print(f"Exported {output}")
    print(f"Source SHA-256: {source_sha256}")
    print(f"Output SHA-256: {output_sha256}")
    print(
        "State/action tensors: "
        f"{len(state_encoder)}/{len(action_encoder)}; dimension=768"
    )


if __name__ == "__main__":
    main()
