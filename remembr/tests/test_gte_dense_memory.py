import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

if "langchain_huggingface" not in sys.modules:
    langchain_huggingface = types.ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = object
    sys.modules["langchain_huggingface"] = langchain_huggingface

from remembr.memory.gte_dense_memory import GteDenseMemory
from remembr.memory.memory import MemoryItem


class FakeGteEncoder:
    def __init__(self):
        self.caption_encode_calls = 0

    def encode_query(self, text):
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def encode_captions(self, texts):
        self.caption_encode_calls += 1
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, 0] = float(index + 1)
        return vectors


def make_memory():
    memory = GteDenseMemory.__new__(GteDenseMemory)
    memory.encoder = FakeGteEncoder()
    memory.embedding_model = "fake-gte"
    memory.model_fingerprint = "fake-model-fingerprint"
    memory.time_offset = 1_672_000_000
    memory.text_k = 2
    memory.numeric_k = 2
    memory.reset()
    for index, score in enumerate([1.0, 3.0, 2.0]):
        item = MemoryItem(
            caption=f"caption {index}",
            time=1_672_000_000 + index,
            position=[float(index), 0.0, 0.0],
            theta=0.0,
        )
        item.entry_id = 100 + index
        embedding = np.zeros(768, dtype=np.float32)
        embedding[0] = score
        memory.insert(item, text_embedding=embedding)
    return memory


class GteDenseMemoryTest(unittest.TestCase):
    def test_text_search_ranks_cosine_similarity_descending(self):
        memory = make_memory()

        returned = memory.search_by_text("query")
        trace = memory.get_retrieval_trace()[0]

        self.assertEqual(
            [record["entry_id"] for record in trace["selected"]],
            [101, 102],
        )
        self.assertEqual(
            [record["entry_id"] for record in trace["ranking"]],
            [101, 102, 100],
        )
        self.assertFalse(trace["lower_is_better"])
        self.assertEqual(trace["embedding_dimension"], 768)
        self.assertIn("caption 1", returned)

    def test_caption_cache_is_hashed_validated_and_reused(self):
        memory = GteDenseMemory.__new__(GteDenseMemory)
        memory.encoder = FakeGteEncoder()
        memory.embedding_model = "fake-gte"
        memory.model_fingerprint = "fake-model-fingerprint"
        memory._sequence_embedding_cache = {}

        captions = [
            {"caption": "first caption"},
            {"caption": "second caption"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            captions_path = os.path.join(directory, "captions.json")
            cache_dir = os.path.join(directory, "cache")
            with open(captions_path, "w") as stream:
                json.dump(captions, stream)

            first = memory.caption_embeddings(
                captions_path,
                captions,
                cache_dir=cache_dir,
            )
            second = memory.caption_embeddings(
                captions_path,
                captions,
                cache_dir=cache_dir,
            )

            self.assertEqual(first.shape, (2, 768))
            np.testing.assert_array_equal(first, second)
            self.assertEqual(memory.encoder.caption_encode_calls, 1)
            self.assertEqual(
                len([name for name in os.listdir(cache_dir) if name.endswith(".npz")]),
                1,
            )


if __name__ == "__main__":
    unittest.main()
