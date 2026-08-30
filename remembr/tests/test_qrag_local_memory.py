import json
import sys
import types
import unittest

import numpy as np

if "langchain_huggingface" not in sys.modules:
    langchain_huggingface = types.ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = object
    sys.modules["langchain_huggingface"] = langchain_huggingface

from remembr.memory.memory import MemoryItem
from remembr.memory.qrag_local_memory import QragLocalMemory
from remembr.memory.qrag_text_retriever import (
    EXPECTED_QRAG_COMMIT,
    EXPECTED_SOURCE_SHA256,
)


class FakeQragEncoder:
    def __init__(self):
        self.encode_calls = 0

    def encode_state(self, components):
        self.encode_calls += 1
        # First choose using the question axis; after one caption has entered
        # state, change policy direction so a static top-k implementation fails.
        if len(components) == 1:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


def make_memory(state_format="native", retrieval_mode="sequential"):
    memory = QragLocalMemory.__new__(QragLocalMemory)
    memory.encoder = FakeQragEncoder()
    memory.embedding_model = "fake-gte"
    memory.source_checkpoint = "fake-checkpoint"
    memory.inference_checkpoint = "fake-inference-checkpoint"
    memory.source_checkpoint_sha256 = EXPECTED_SOURCE_SHA256
    memory.time_offset = 1_672_000_000
    memory.text_k = 2
    memory.numeric_k = 2
    memory.state_format = state_format
    memory.retrieval_mode = retrieval_mode
    memory.original_question = None
    memory.reset()
    memory.set_qrag_context("Which evidence answers the question?")

    embeddings = [[3.0, 0.0], [2.0, 1.0], [1.0, 4.0]]
    for index, embedding in enumerate(embeddings):
        item = MemoryItem(
            caption=f"caption {index}",
            time=1_672_000_000 + index,
            position=[float(index), 0.0, 0.0],
            theta=0.0,
        )
        item.entry_id = 100 + index
        memory.insert(item, text_embedding=embedding)
    return memory


class QragLocalMemoryTest(unittest.TestCase):
    def test_static_topk_encodes_question_once_and_does_not_update_state(self):
        memory = make_memory(retrieval_mode="static")

        returned = memory.search_by_text("controller tool query")
        trace = memory.get_retrieval_trace()[0]

        self.assertEqual(
            [record["entry_id"] for record in trace["selected"]],
            [100, 101],
        )
        self.assertEqual(memory.encoder.encode_calls, 1)
        self.assertEqual(trace["state_components"], [
            "Which evidence answers the question?"
        ])
        self.assertEqual(trace["steps"], [])
        self.assertEqual(trace["retrieval_method"], "qrag_static_topk_zero_shot")
        self.assertEqual(trace["qrag_selection_mode"], "static_topk")
        self.assertEqual(trace["state_encode_count"], 1)
        self.assertIn("caption 1", returned)
        json.dumps(trace)

    def test_sequential_policy_reencodes_state_and_masks_selected_action(self):
        memory = make_memory()

        returned = memory.search_by_text("controller tool query")
        trace = memory.get_retrieval_trace()[0]

        self.assertEqual(
            [record["entry_id"] for record in trace["selected"]],
            [100, 102],
        )
        self.assertEqual(
            [record["entry_id"] for record in trace["ranking"]],
            [100, 101, 102],
        )
        self.assertEqual(len(trace["steps"]), 2)
        self.assertEqual(trace["steps"][1]["state_components"][-1], "caption 0")
        self.assertNotIn(
            100,
            [record["entry_id"] for record in trace["steps"][1]["ranking"]],
        )
        self.assertEqual(trace["retrieval_method"], "qrag_sequential_zero_shot")
        self.assertEqual(trace["source_checkpoint_sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(trace["qrag_code_commit"], EXPECTED_QRAG_COMMIT)
        self.assertFalse(trace["lower_is_better"])
        self.assertIn("caption 2", returned)
        json.dumps(trace)

    def test_controller_state_includes_tool_query_after_original_question(self):
        memory = make_memory(state_format="controller")
        memory.search_by_text("controller tool query")
        first_state = memory.get_retrieval_trace()[0]["steps"][0][
            "state_components"
        ]
        self.assertEqual(
            first_state,
            ["Which evidence answers the question?", "controller tool query"],
        )


if __name__ == "__main__":
    unittest.main()
