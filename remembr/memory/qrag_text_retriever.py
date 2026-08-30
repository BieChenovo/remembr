"""Slim, inference-only adapter for the official Q-RAG GTE checkpoint."""

import math
import os

import numpy as np


EXPECTED_SOURCE_SHA256 = (
    "ff2ab1db095fe05f0f854672a224e9dff6ff0a9a8ecda5cf35cdf88a94d37c56"
)
EXPECTED_QRAG_COMMIT = "42358d78ac491843763b90677f07237471c97086"


class QragEncoderBundle:
    """Shared state/action encoders reproducing Q-RAG ``BertPredictor``.

    The upstream module defines a 768->384 ``head`` but does not call it in
    ``forward``. It attention-mask mean-pools the 768-dimensional transformer
    output and divides by ten. The adapter intentionally reproduces that actual
    behavior instead of the stale 384-dimensional description in the spec.
    """

    _instances = {}

    def __new__(
        cls,
        model_name,
        inference_checkpoint,
        device="cpu",
        batch_size=16,
    ):
        key = (
            os.path.realpath(model_name),
            os.path.realpath(inference_checkpoint),
            str(device),
            int(batch_size),
        )
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        model_name,
        inference_checkpoint,
        device="cpu",
        batch_size=16,
    ):
        if self._initialized:
            return

        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model_name = os.path.realpath(model_name)
        self.inference_checkpoint = os.path.realpath(inference_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        payload = torch.load(
            self.inference_checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if payload.get("version") != 1:
            raise ValueError(
                f"Unsupported Q-RAG inference checkpoint version: "
                f"{payload.get('version')!r}"
            )
        metadata = payload.get("metadata", {})
        if metadata.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError("Q-RAG source checkpoint fingerprint mismatch")
        if metadata.get("qrag_code_commit") != EXPECTED_QRAG_COMMIT:
            raise ValueError("Q-RAG source-code commit mismatch")
        if metadata.get("embedding_dimension") != 768:
            raise ValueError("Q-RAG inference checkpoint must output 768 dimensions")
        if metadata.get("positions_processor") != "none":
            raise ValueError("This adapter only supports positions_processor=none")
        self.metadata = dict(metadata)

        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        config.num_hidden_layers = 12
        self.state_model = AutoModel.from_config(
            config,
            trust_remote_code=True,
        )
        self.action_model = AutoModel.from_config(
            config,
            trust_remote_code=True,
        )
        self._load_predictor_model(
            self.state_model,
            payload["state_encoder"],
            "state",
        )
        self._load_predictor_model(
            self.action_model,
            payload["action_encoder"],
            "action",
        )
        del payload

        self.state_model.to(self.device).eval()
        self.action_model.to(self.device).eval()
        self._initialized = True

    @staticmethod
    def _load_predictor_model(model, predictor_state, name):
        model_state = {
            key[len("model.") :]: value
            for key, value in predictor_state.items()
            if key.startswith("model.")
        }
        if len(model_state) != 136:
            raise ValueError(
                f"Expected 136 {name} transformer tensors, got {len(model_state)}"
            )
        incompatible = model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Could not load exact Q-RAG {name} encoder: {incompatible}"
            )

    def _mean_div_10(self, model, input_ids, attention_mask):
        with self.torch.inference_mode():
            hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
            mask = attention_mask.reshape(hidden.shape[0], hidden.shape[1], 1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = pooled / 10.0
        return pooled.float().cpu().numpy()

    @staticmethod
    def _next_power_of_two(value):
        return 1 if value <= 1 else 2 ** int(math.ceil(math.log2(value)))

    def encode_actions(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            unpadded = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
            )
            max_tokens = max(len(ids) for ids in unpadded["input_ids"])
            padded_length = self._next_power_of_two(max_tokens)
            batch = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding="max_length",
                # Upstream ``custom_pad_sequence`` pads each batch to the next
                # power of two, including its attention mask.
                return_tensors="pt",
            )
            if padded_length < 256:
                token_slice = (
                    slice(-padded_length, None)
                    if self.tokenizer.padding_side == "left"
                    else slice(None, padded_length)
                )
                batch = {
                    key: value[:, token_slice]
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
            else:
                batch = {
                    key: value
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs.append(
                self._mean_div_10(
                    self.action_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
            )
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def _stack_state_components(self, components):
        tokens_split = self.tokenizer(
            list(components),
            truncation=True,
            max_length=256,
        )
        input_sequences = tokens_split["input_ids"]
        mask_sequences = tokens_split["attention_mask"]
        sep_token = self.tokenizer.sep_token_id
        if sep_token is None:
            sep_token = self.tokenizer.eos_token_id
        combined_ids = list(input_sequences[0])
        combined_mask = list(mask_sequences[0])
        for next_ids, next_mask in zip(
            input_sequences[1:],
            mask_sequences[1:],
        ):
            combined_ids = combined_ids[:-1] + [sep_token] + list(next_ids[1:])
            combined_mask = combined_mask[:-1] + [1] + list(next_mask[1:])

        padded_length = self._next_power_of_two(len(combined_ids))
        pad_count = padded_length - len(combined_ids)
        if self.tokenizer.padding_side == "left":
            combined_ids = [self.tokenizer.pad_token_id] * pad_count + combined_ids
            combined_mask = [0] * pad_count + combined_mask
        else:
            combined_ids += [self.tokenizer.pad_token_id] * pad_count
            combined_mask += [0] * pad_count
        return (
            self.torch.tensor(
                [combined_ids],
                dtype=self.torch.long,
                device=self.device,
            ),
            self.torch.tensor(
                [combined_mask],
                dtype=self.torch.long,
                device=self.device,
            ),
        )

    def encode_state(self, components):
        components = [str(component) for component in components]
        if not components:
            raise ValueError("Q-RAG state must contain at least the question")
        input_ids, attention_mask = self._stack_state_components(components)
        return self._mean_div_10(
            self.state_model,
            input_ids,
            attention_mask,
        )[0]
