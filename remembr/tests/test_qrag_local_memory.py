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


def make_memory(
    state_format="native",
    retrieval_mode="sequential",
    episode_mode="per_call",
    question_evidence_budget=2,
):
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
    memory.episode_mode = episode_mode
    memory.question_evidence_budget = question_evidence_budget
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

    def test_question_episode_carries_state_masks_ids_and_exhausts_budget(self):
        memory = make_memory(
            state_format="controller",
            episode_mode="question",
            question_evidence_budget=3,
        )
        memory.begin_retrieval_episode()

        memory.search_by_text("first tool query")
        self.assertTrue(memory.text_retrieval_available())
        memory.search_by_text("different second query")
        self.assertFalse(memory.text_retrieval_available())
        exhausted_context = memory.search_by_text("third query")
        traces = memory.get_retrieval_trace()

        self.assertEqual(
            [record["entry_id"] for record in traces[0]["selected"]],
            [102, 101],
        )
        self.assertEqual(
            [record["entry_id"] for record in traces[1]["selected"]],
            [100],
        )
        self.assertEqual(traces[1]["episode_selected_entry_ids_before"], [102, 101])
        self.assertEqual(
            traces[1]["steps"][0]["state_components"],
            [
                "Which evidence answers the question?",
                "different second query",
                "caption 2",
                "caption 1",
            ],
        )
        self.assertEqual(traces[2]["selected"], [])
        self.assertEqual(traces[2]["effective_requested_k"], 0)
        self.assertTrue(traces[2]["budget_exhausted"])
        self.assertIn("evidence budget is exhausted", exhausted_context)
        selected_ids = [
            record["entry_id"]
            for trace in traces
            for record in trace["selected"]
        ]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))

    def test_new_answer_attempt_resets_question_episode_not_trace(self):
        memory = make_memory(
            state_format="controller",
            episode_mode="question",
            question_evidence_budget=2,
        )
        memory.begin_retrieval_episode()
        memory.search_by_text("first attempt")
        first_trace = memory.get_retrieval_trace()[0]

        memory.begin_retrieval_episode()
        memory.search_by_text("retry attempt")
        traces = memory.get_retrieval_trace()

        self.assertEqual(len(traces), 2)
        self.assertEqual(first_trace["qrag_episode_id"], 1)
        self.assertEqual(traces[1]["qrag_episode_id"], 2)
        self.assertEqual(traces[1]["episode_selected_entry_ids_before"], [])
        self.assertEqual(
            [record["entry_id"] for record in traces[0]["selected"]],
            [record["entry_id"] for record in traces[1]["selected"]],
        )

    def test_static_question_episode_uses_query_and_global_mask(self):
        memory = make_memory(
            state_format="controller",
            retrieval_mode="static",
            episode_mode="question",
            question_evidence_budget=3,
        )
        memory.begin_retrieval_episode()

        memory.search_by_text("first query")
        memory.search_by_text("second query")
        traces = memory.get_retrieval_trace()

        self.assertEqual(
            [record["entry_id"] for record in traces[0]["selected"]],
            [102, 101],
        )
        self.assertEqual(
            [record["entry_id"] for record in traces[1]["selected"]],
            [100],
        )
        self.assertEqual(
            traces[1]["state_components"],
            ["Which evidence answers the question?", "second query"],
        )
        self.assertEqual(traces[1]["episode_selected_entry_ids_before"], [102, 101])


if __name__ == "__main__":
    unittest.main()
