"""ReMEmbR local memory with Q-RAG sequential text retrieval."""

import fcntl
import hashlib
import json
import os
import re
import tempfile

import numpy as np

from remembr.memory.local_vector_memory import LocalVectorMemory
from remembr.memory.qrag_text_retriever import (
    EXPECTED_QRAG_COMMIT,
    EXPECTED_SOURCE_SHA256,
    QragEncoderBundle,
)


class QragLocalMemory(LocalVectorMemory):
    """Replace only text search; inherit position/time behavior unchanged."""

    ENCODER_SIGNATURE = "qrag_gte_mean_div10_sequential_v1"
    _sequence_embedding_cache = {}

    def __init__(
        self,
        model_name,
        source_checkpoint,
        inference_checkpoint,
        time_offset,
        evidence_budget=5,
        numeric_k=4,
        state_format="controller",
        retrieval_mode="sequential",
        episode_mode="question",
        question_evidence_budget=None,
        device="cpu",
        batch_size=16,
    ):
        if state_format not in {"native", "controller"}:
            raise ValueError(f"Unsupported Q-RAG state format: {state_format}")
        if retrieval_mode not in {"static", "sequential"}:
            raise ValueError(f"Unsupported Q-RAG retrieval mode: {retrieval_mode}")
        if episode_mode not in {"per_call", "question"}:
            raise ValueError(f"Unsupported Q-RAG episode mode: {episode_mode}")
        if question_evidence_budget is not None and int(question_evidence_budget) < 1:
            raise ValueError("Q-RAG question evidence budget must be positive")
        self.embedding_model = os.path.realpath(model_name)
        self.source_checkpoint = os.path.realpath(source_checkpoint)
        self.inference_checkpoint = os.path.realpath(inference_checkpoint)
        self.source_checkpoint_sha256 = self._read_source_fingerprint()
        self.encoder = QragEncoderBundle(
            model_name=self.embedding_model,
            inference_checkpoint=self.inference_checkpoint,
            device=device,
            batch_size=batch_size,
        )
        self.time_offset = time_offset
        self.text_k = int(evidence_budget)
        self.numeric_k = int(numeric_k)
        self.state_format = state_format
        self.retrieval_mode = retrieval_mode
        self.episode_mode = episode_mode
        self.question_evidence_budget = int(
            question_evidence_budget
            if question_evidence_budget is not None
            else evidence_budget
        )
        self.original_question = None
        self.reset()

    def reset(self):
        super().reset()
        self._qrag_episode_id = 0
        self._qrag_text_call_count = 0
        self._qrag_episode_selected_indices = []

    def begin_retrieval_episode(self):
        """Start an independent answer attempt without discarding trace history."""

        self._qrag_episode_id += 1
        self._qrag_text_call_count = 0
        self._qrag_episode_selected_indices = []

    def text_retrieval_available(self):
        if not self.items:
            return False
        if self.episode_mode != "question":
            return True
        return (
            len(self._qrag_episode_selected_indices) < self.question_evidence_budget
            and len(self._qrag_episode_selected_indices) < len(self.items)
        )

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_source_fingerprint(self):
        sidecar = f"{self.source_checkpoint}.sha256"
        if not os.path.isfile(sidecar):
            raise ValueError(f"Missing Q-RAG checkpoint digest sidecar: {sidecar}")
        with open(sidecar, "r") as stream:
            fingerprint = stream.read().strip().split()[0]
        if fingerprint != EXPECTED_SOURCE_SHA256:
            raise ValueError(
                f"Unexpected Q-RAG source checkpoint digest: {fingerprint}"
            )
        return fingerprint

    def set_qrag_context(self, original_question):
        self.original_question = str(original_question)

    @staticmethod
    def _model_slug(model_name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")[-96:]

    def _cache_path(self, captions_path, cache_dir, caption_sha256):
        filename = os.path.splitext(os.path.basename(captions_path))[0]
        slug = self._model_slug(self.embedding_model)
        return os.path.join(
            cache_dir,
            f"{filename}__{slug}__{self.source_checkpoint_sha256[:16]}__"
            f"{self.ENCODER_SIGNATURE}__{caption_sha256[:16]}.npz",
        )

    def _read_cache(self, cache_path, caption_sha256, entry_count):
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            entry_ids = np.asarray(payload["entry_ids"], dtype=np.int64)
        expected_ids = np.arange(entry_count, dtype=np.int64)
        expected = {
            "version": 1,
            "caption_file_sha256": caption_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "qrag_code_commit": EXPECTED_QRAG_COMMIT,
            "encoder_signature": self.ENCODER_SIGNATURE,
            "embedding_dimension": 768,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Q-RAG cache metadata mismatch for {key}: {cache_path}"
                )
        if not np.array_equal(entry_ids, expected_ids):
            raise ValueError(f"Q-RAG cache entry order mismatch: {cache_path}")
        if embeddings.shape != (entry_count, 768):
            raise ValueError(
                f"Invalid Q-RAG action cache shape {embeddings.shape}: {cache_path}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Non-finite Q-RAG action cache: {cache_path}")
        return embeddings

    def caption_embeddings(self, captions_path, captions, cache_dir=None):
        key = (
            os.path.realpath(captions_path),
            self.source_checkpoint_sha256,
            self.ENCODER_SIGNATURE,
            os.path.realpath(cache_dir) if cache_dir else None,
        )
        if key in self._sequence_embedding_cache:
            return self._sequence_embedding_cache[key]
        caption_sha256 = self._sha256(captions_path)
        texts = [item["caption"] for item in captions]
        if cache_dir is None:
            embeddings = self.encoder.encode_actions(texts)
            self._sequence_embedding_cache[key] = embeddings
            return embeddings

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = self._cache_path(
            captions_path,
            cache_dir,
            caption_sha256,
        )
        with open(f"{cache_path}.lock", "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(cache_path):
                embeddings = self._read_cache(
                    cache_path,
                    caption_sha256,
                    len(captions),
                )
            else:
                embeddings = self.encoder.encode_actions(texts)
                if embeddings.shape != (len(captions), 768):
                    raise ValueError(
                        f"Q-RAG produced {embeddings.shape}, expected "
                        f"{(len(captions), 768)}"
                    )
                metadata = {
                    "version": 1,
                    "caption_file": os.path.realpath(captions_path),
                    "caption_file_sha256": caption_sha256,
                    "source_checkpoint_sha256": self.source_checkpoint_sha256,
                    "qrag_code_commit": EXPECTED_QRAG_COMMIT,
                    "encoder": "Alibaba-NLP/gte-multilingual-base",
                    "encoder_signature": self.ENCODER_SIGNATURE,
                    "entry_count": len(captions),
                    "embedding_shape": list(embeddings.shape),
                    "embedding_dimension": 768,
                    "dtype": "float32",
                    "pooling": "attention_masked_mean_div_10",
                    "positions_processor": "none",
                }
                temporary = tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix=".qrag-action-cache-",
                    suffix=".npz",
                    delete=False,
                )
                temporary_path = temporary.name
                temporary.close()
                try:
                    np.savez_compressed(
                        temporary_path,
                        embeddings=embeddings.astype(np.float32),
                        entry_ids=np.arange(len(captions), dtype=np.int64),
                        metadata_json=np.asarray(
                            json.dumps(metadata, sort_keys=True)
                        ),
                    )
                    os.replace(temporary_path, cache_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.unlink(temporary_path)
        self._sequence_embedding_cache[key] = embeddings
        return embeddings

    def insert(self, item, text_embedding=None):
        if text_embedding is None:
            text_embedding = self.encoder.encode_actions([item.caption])[0]
        self.items.append(item)
        self.text_embeddings.append(np.asarray(text_embedding, dtype=np.float32))

    def _episode_prior_indices(self):
        if self.episode_mode != "question":
            return []
        return list(self._qrag_episode_selected_indices)

    def _remaining_question_budget(self, prior_indices):
        if self.episode_mode != "question":
            return None
        return max(self.question_evidence_budget - len(prior_indices), 0)

    def _requested_count(self, candidate_count, prior_indices):
        available_count = max(candidate_count - len(set(prior_indices)), 0)
        requested = min(self.text_k, available_count)
        remaining = self._remaining_question_budget(prior_indices)
        if remaining is not None:
            requested = min(requested, remaining)
        return requested, remaining

    def _state_components(
        self,
        tool_query,
        selected_indices,
        *,
        include_episode_evidence=True,
    ):
        if self.original_question is None:
            raise ValueError(
                "Q-RAG original question was not set before text retrieval"
            )
        components = [self.original_question]
        if self.state_format == "controller":
            components.append(tool_query)
        evidence_indices = []
        if include_episode_evidence:
            evidence_indices.extend(self._episode_prior_indices())
        evidence_indices.extend(selected_indices)
        components.extend(self.items[index].caption for index in evidence_indices)
        return components

    def _commit_episode_selection(self, selected_indices):
        if self.episode_mode != "question":
            return
        seen = set(self._qrag_episode_selected_indices)
        for index in selected_indices:
            if index not in seen:
                self._qrag_episode_selected_indices.append(index)
                seen.add(index)

    def _episode_trace_fields(self, prior_indices, selected_indices):
        after_indices = list(prior_indices)
        after_seen = set(after_indices)
        for index in selected_indices:
            if index not in after_seen:
                after_indices.append(index)
                after_seen.add(index)
        remaining_before = self._remaining_question_budget(prior_indices)
        remaining_after = self._remaining_question_budget(after_indices)
        return {
            "qrag_episode_mode": self.episode_mode,
            "qrag_episode_id": self._qrag_episode_id,
            "qrag_text_call_index": self._qrag_text_call_count,
            "question_evidence_budget": (
                self.question_evidence_budget
                if self.episode_mode == "question"
                else None
            ),
            "question_budget_remaining_before": remaining_before,
            "question_budget_remaining_after": remaining_after,
            "episode_selected_entry_ids_before": [
                getattr(self.items[index], "entry_id", None)
                for index in prior_indices
            ],
            "episode_selected_entry_ids_after": [
                getattr(self.items[index], "entry_id", None)
                for index in after_indices
            ],
            "budget_exhausted": (
                remaining_after == 0 if remaining_after is not None else False
            ),
        }

    @staticmethod
    def _empty_result_context(budget_exhausted):
        if budget_exhausted:
            return (
                "No additional text memories were retrieved because the "
                "question-level evidence budget is exhausted."
            )
        return "No additional text memories were available."

    def _search_by_text_static(self, query: str) -> str:
        candidate_count = len(self.items)
        self._qrag_text_call_count += 1
        prior_indices = self._episode_prior_indices()
        requested, _ = self._requested_count(candidate_count, prior_indices)
        components = self._state_components(
            query,
            [],
            # Static mode keeps a fixed question/query state across ranking
            # calls; only its global selected-ID mask advances.
            include_episode_evidence=False,
        )
        action_embeddings = np.asarray(self.text_embeddings, dtype=np.float32)
        available = np.ones(candidate_count, dtype=bool)
        if prior_indices:
            available[np.asarray(prior_indices, dtype=int)] = False
        if candidate_count and requested:
            state_embedding = np.asarray(
                self.encoder.encode_state(components),
                dtype=np.float32,
            )
            scores = action_embeddings @ state_embedding
            available_indices = np.flatnonzero(available)
            ranking_indices = available_indices[
                np.argsort(-scores[available_indices], kind="stable")
            ]
        else:
            scores = np.zeros(candidate_count, dtype=np.float32)
            ranking_indices = np.asarray([], dtype=np.int64)
        selected_indices = [int(index) for index in ranking_indices[:requested]]
        ranking = [
            {
                "rank": rank,
                "entry_id": getattr(self.items[int(index)], "entry_id", None),
                "local_index": int(index),
                "score": float(scores[int(index)]),
            }
            for rank, index in enumerate(ranking_indices, start=1)
        ]
        selected_records = []
        for rank, index in enumerate(selected_indices, start=1):
            record = self._item_record(self.items[index], index)
            record.update({"rank": rank, "score": float(scores[index])})
            selected_records.append(record)

        selected = [self.items[index] for index in selected_indices]
        self.working_memory.extend(selected)
        self._commit_episode_selection(selected_indices)
        episode_fields = self._episode_trace_fields(
            prior_indices,
            selected_indices,
        )
        trace = {
            "call_index": len(self.retrieval_trace) + 1,
            "tool": "retrieve_from_text",
            "query": query,
            "parsed_query": query,
            "original_question": self.original_question,
            "state_components": list(components),
            "score_name": "qrag_question_action_dot_product",
            "score_unit": None,
            "lower_is_better": False,
            "candidate_count": candidate_count,
            "requested_k": self.text_k,
            "effective_requested_k": requested,
            "returned_count": len(selected),
            "selected": selected_records,
            "ranking": ranking,
            "steps": [],
            "retrieval_method": "qrag_static_topk_zero_shot",
            "qrag_selection_mode": "static_topk",
            "qrag_state_format": self.state_format,
            "state_encode_count": 1 if requested else 0,
            "embedding_model": "Alibaba-NLP/gte-multilingual-base",
            "embedding_dimension": 768,
            "pooling": "attention_masked_mean_div_10",
            "positions_processor": "none",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "qrag_code_commit": EXPECTED_QRAG_COMMIT,
            "training_datasets": ["HotpotQA", "Musique"],
            "training_max_steps": 6,
            "evidence_budget": self.text_k,
        }
        trace.update(episode_fields)
        output = (
            self.memory_to_string(selected)
            if selected
            else self._empty_result_context(trace["budget_exhausted"])
        )
        trace["returned_context"] = output
        self.retrieval_trace.append(trace)
        return output

    def _search_by_text_sequential(self, query: str) -> str:
        candidate_count = len(self.items)
        self._qrag_text_call_count += 1
        prior_indices = self._episode_prior_indices()
        requested, _ = self._requested_count(candidate_count, prior_indices)
        action_embeddings = np.asarray(self.text_embeddings, dtype=np.float32)
        available = np.ones(candidate_count, dtype=bool)
        if prior_indices:
            available[np.asarray(prior_indices, dtype=int)] = False
        selected_indices = []
        selected_records = []
        steps = []
        initial_ranking = []

        for step_index in range(requested):
            components = self._state_components(query, selected_indices)
            state_embedding = np.asarray(
                self.encoder.encode_state(components),
                dtype=np.float32,
            )
            scores = action_embeddings @ state_embedding
            available_indices = np.flatnonzero(available)
            available_ranking = available_indices[
                np.argsort(-scores[available_indices], kind="stable")
            ]
            selected_index = int(available_ranking[0])
            if step_index == 0:
                initial_ranking = [
                    {
                        "rank": rank,
                        "entry_id": getattr(self.items[int(index)], "entry_id", None),
                        "local_index": int(index),
                        "score": float(scores[int(index)]),
                    }
                    for rank, index in enumerate(available_ranking, start=1)
                ]

            record = self._item_record(
                self.items[selected_index],
                selected_index,
            )
            record.update(
                {
                    "rank": step_index + 1,
                    "score": float(scores[selected_index]),
                    "selection_step": step_index + 1,
                }
            )
            selected_records.append(record)
            steps.append(
                {
                    "step": step_index + 1,
                    "state_components": list(components),
                    "selected_entry_id": record["entry_id"],
                    "selected_local_index": selected_index,
                    "selected_score": float(scores[selected_index]),
                    "available_candidate_count": len(available_ranking),
                    "stored_ranking_count": min(20, len(available_ranking)),
                    "ranking_truncated": len(available_ranking) > 20,
                    "ranking": [
                        {
                            "rank": rank,
                            "entry_id": getattr(
                                self.items[int(index)],
                                "entry_id",
                                None,
                            ),
                            "local_index": int(index),
                            "score": float(scores[int(index)]),
                        }
                        for rank, index in enumerate(
                            available_ranking[:20],
                            start=1,
                        )
                    ],
                }
            )
            selected_indices.append(selected_index)
            available[selected_index] = False

        selected = [self.items[index] for index in selected_indices]
        self.working_memory.extend(selected)
        self._commit_episode_selection(selected_indices)
        episode_fields = self._episode_trace_fields(
            prior_indices,
            selected_indices,
        )
        trace = {
            "call_index": len(self.retrieval_trace) + 1,
            "tool": "retrieve_from_text",
            "query": query,
            "parsed_query": query,
            "original_question": self.original_question,
            "score_name": "qrag_policy_state_action_dot_product",
            "score_unit": None,
            "lower_is_better": False,
            "candidate_count": candidate_count,
            "requested_k": self.text_k,
            "effective_requested_k": requested,
            "returned_count": len(selected),
            "selected": selected_records,
            "ranking": initial_ranking,
            "steps": steps,
            "retrieval_method": "qrag_sequential_zero_shot",
            "qrag_selection_mode": "sequential",
            "qrag_state_format": self.state_format,
            "state_encode_count": len(steps),
            "embedding_model": "Alibaba-NLP/gte-multilingual-base",
            "embedding_dimension": 768,
            "pooling": "attention_masked_mean_div_10",
            "positions_processor": "none",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "qrag_code_commit": EXPECTED_QRAG_COMMIT,
            "training_datasets": ["HotpotQA", "Musique"],
            "training_max_steps": 6,
            "inference_fixed_steps": self.text_k,
        }
        trace.update(episode_fields)
        output = (
            self.memory_to_string(selected)
            if selected
            else self._empty_result_context(trace["budget_exhausted"])
        )
        trace["returned_context"] = output
        self.retrieval_trace.append(trace)
        return output

    def search_by_text(self, query: str) -> str:
        if self.retrieval_mode == "static":
            return self._search_by_text_static(query)
        return self._search_by_text_sequential(query)
