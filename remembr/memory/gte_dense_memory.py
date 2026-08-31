"""GTE-base dense retrieval for the B1 NaVQA ablation.

This module deliberately does not load Q-RAG fine-tuning weights.  It uses the
base model's documented dense-retrieval interface: the first-token embedding,
L2 normalized, followed by cosine similarity (a dot product after
normalization). Caption vectors are cached per complete sequence so question
windows never encode the same caption repeatedly.
"""

import fcntl
import glob
import hashlib
import json
import os
import re
import tempfile

import numpy as np

from remembr.memory.local_vector_memory import LocalVectorMemory


class GteDenseEncoder:
    """Lazy, shared wrapper around the unfine-tuned GTE base encoder."""

    _instances = {}

    def __new__(cls, model_name, device="cpu", batch_size=16):
        key = (os.path.realpath(model_name), device, int(batch_size))
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, model_name, device="cpu", batch_size=16):
        if self._initialized:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self._initialized = True

    def encode(self, texts, max_length):
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                pooled = hidden[:, 0]
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def encode_query(self, text):
        return self.encode([text], max_length=256)[0]

    def encode_captions(self, texts):
        return self.encode(texts, max_length=512)


class GteDenseMemory(LocalVectorMemory):
    """Local memory whose text view uses unfine-tuned GTE dot-product search."""

    ENCODER_SIGNATURE = "gte_first_token_l2_v1"
    _model_fingerprint_cache = {}
    _sequence_embedding_cache = {}

    def __init__(
        self,
        model_name,
        time_offset,
        text_k=5,
        numeric_k=4,
        text_episode_mode="per_call",
        question_text_evidence_budget=None,
        device="cpu",
        batch_size=16,
    ):
        if text_episode_mode not in {"per_call", "question"}:
            raise ValueError(
                f"Unsupported text retrieval episode mode: {text_episode_mode}"
            )
        if (
            question_text_evidence_budget is not None
            and int(question_text_evidence_budget) < 1
        ):
            raise ValueError("Question text evidence budget must be positive")
        self.encoder = GteDenseEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        self.embedding_model = model_name
        self.model_fingerprint = self._model_fingerprint(model_name)
        self.time_offset = time_offset
        self.text_k = text_k
        self.numeric_k = numeric_k
        self.text_episode_mode = text_episode_mode
        self.question_text_evidence_budget = int(
            question_text_evidence_budget
            if question_text_evidence_budget is not None
            else text_k
        )
        self.reset()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _model_slug(model_name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")[-96:]

    @classmethod
    def _model_fingerprint(cls, model_name):
        resolved = os.path.realpath(model_name)
        if resolved in cls._model_fingerprint_cache:
            return cls._model_fingerprint_cache[resolved]

        digest = hashlib.sha256()
        if os.path.isdir(resolved):
            candidates = [os.path.join(resolved, "config.json")]
            candidates.extend(glob.glob(os.path.join(resolved, "*.safetensors")))
            candidates.extend(glob.glob(os.path.join(resolved, "pytorch_model*.bin")))
            candidates = sorted(path for path in candidates if os.path.isfile(path))
            if not candidates:
                raise ValueError(f"No model config or weights found in {resolved}")
            for path in candidates:
                digest.update(os.path.basename(path).encode("utf-8"))
                with open(path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        else:
            digest.update(model_name.encode("utf-8"))

        fingerprint = digest.hexdigest()
        cls._model_fingerprint_cache[resolved] = fingerprint
        return fingerprint

    def _cache_path(self, captions_path, cache_dir, caption_sha256):
        filename = os.path.splitext(os.path.basename(captions_path))[0]
        slug = self._model_slug(self.embedding_model)
        return os.path.join(
            cache_dir,
            f"{filename}__{slug}__{self.model_fingerprint[:16]}__"
            f"{self.ENCODER_SIGNATURE}__{caption_sha256[:16]}.npz",
        )

    def _read_cache(self, cache_path, caption_sha256, entry_count):
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            entry_ids = np.asarray(payload["entry_ids"], dtype=np.int64)

        expected_ids = np.arange(entry_count, dtype=np.int64)
        if metadata.get("version") != 1:
            raise ValueError(f"Unsupported GTE cache version in {cache_path}")
        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError(f"GTE cache model mismatch in {cache_path}")
        if metadata.get("model_fingerprint") != self.model_fingerprint:
            raise ValueError(f"GTE cache model fingerprint mismatch in {cache_path}")
        if metadata.get("caption_file_sha256") != caption_sha256:
            raise ValueError(f"GTE cache caption hash mismatch in {cache_path}")
        if metadata.get("encoder_signature") != self.ENCODER_SIGNATURE:
            raise ValueError(f"GTE cache encoder signature mismatch in {cache_path}")
        if not np.array_equal(entry_ids, expected_ids):
            raise ValueError(f"GTE cache entry order mismatch in {cache_path}")
        if embeddings.shape[0] != entry_count or embeddings.ndim != 2:
            raise ValueError(f"Invalid GTE cache shape {embeddings.shape} in {cache_path}")
        if embeddings.shape[1] != 768:
            raise ValueError(
                f"Expected GTE-base vectors with dimension 768, "
                f"got {embeddings.shape[1]} in {cache_path}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"GTE cache contains non-finite values: {cache_path}")
        return embeddings

    def caption_embeddings(self, captions_path, captions, cache_dir=None):
        """Return full-sequence action vectors, building a validated cache once."""
        key = (
            os.path.realpath(captions_path),
            self.embedding_model,
            self.model_fingerprint,
            self.ENCODER_SIGNATURE,
            os.path.realpath(cache_dir) if cache_dir else None,
        )
        if key in self._sequence_embedding_cache:
            return self._sequence_embedding_cache[key]

        caption_sha256 = self._sha256(captions_path)
        texts = [item["caption"] for item in captions]
        if cache_dir is None:
            embeddings = self.encoder.encode_captions(texts)
            self._sequence_embedding_cache[key] = embeddings
            return embeddings

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = self._cache_path(captions_path, cache_dir, caption_sha256)
        lock_path = f"{cache_path}.lock"
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(cache_path):
                embeddings = self._read_cache(
                    cache_path,
                    caption_sha256,
                    len(captions),
                )
            else:
                embeddings = self.encoder.encode_captions(texts)
                if embeddings.shape != (len(captions), 768):
                    raise ValueError(
                        f"GTE produced unexpected caption shape {embeddings.shape}; "
                        f"expected {(len(captions), 768)}"
                    )
                metadata = {
                    "version": 1,
                    "embedding_model": self.embedding_model,
                    "model_fingerprint": self.model_fingerprint,
                    "encoder_signature": self.ENCODER_SIGNATURE,
                    "caption_file": os.path.realpath(captions_path),
                    "caption_file_sha256": caption_sha256,
                    "entry_count": len(captions),
                    "embedding_shape": list(embeddings.shape),
                    "dtype": "float32",
                    "pooling": "first_token_l2_normalized",
                    "score": "cosine_similarity",
                }
                temporary = tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix=".gte-cache-",
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
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                    )
                    os.replace(temporary_path, cache_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.unlink(temporary_path)

        self._sequence_embedding_cache[key] = embeddings
        return embeddings

    def insert(self, item, text_embedding=None):
        if text_embedding is None:
            text_embedding = self.encoder.encode_captions([item.caption])[0]
        self.items.append(item)
        self.text_embeddings.append(np.asarray(text_embedding, dtype=np.float32))

    def search_by_text(self, query: str) -> str:
        query_embedding = np.asarray(
            self.encoder.encode_query(query),
            dtype=np.float32,
        )
        embeddings = np.asarray(self.text_embeddings, dtype=np.float32)
        scores = embeddings @ query_embedding
        selected, trace = self._select(
            scores,
            self.text_k,
            tool="retrieve_from_text",
            query=query,
            parsed_query=query,
            score_name="gte_base_cosine_similarity",
            lower_is_better=False,
        )
        trace["embedding_model"] = self.embedding_model
        trace["embedding_dimension"] = int(query_embedding.shape[0])
        trace["retrieval_method"] = "gte_dense"
        output = self.memory_to_string(selected)
        if not selected:
            output = self.empty_text_result_context(trace["budget_exhausted"])
        trace["returned_context"] = output
        return output
