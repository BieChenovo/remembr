import json
import sys
import types
import unittest

import numpy as np

# The trace tests bypass ``__init__`` and use a fake embedder, so they should not
# require the heavyweight LangChain package merely to import the module.
if "langchain_huggingface" not in sys.modules:
    langchain_huggingface = types.ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = object
    sys.modules["langchain_huggingface"] = langchain_huggingface

from remembr.memory.local_vector_memory import LocalVectorMemory
from remembr.memory.memory import MemoryItem


class FakeEmbedder:
    def embed_query(self, query):
        vectors = {
            "door": [0.0, 0.0],
            "hall": [10.0, 0.0],
        }
        return vectors[query]


def make_memory(
    text_episode_mode="per_call",
    question_text_evidence_budget=2,
    unified_evidence_ledger=False,
    question_evidence_budget=3,
):
    memory = LocalVectorMemory.__new__(LocalVectorMemory)
    memory.embedder = FakeEmbedder()
    memory.time_offset = 1_672_000_000
    memory.text_k = 2
    memory.numeric_k = 2
    memory.text_episode_mode = text_episode_mode
    memory.question_text_evidence_budget = question_text_evidence_budget
    memory.unified_evidence_ledger = unified_evidence_ledger
    memory.question_evidence_budget = question_evidence_budget
    memory.reset()
    memory.set_candidate_pool_metadata(
        {"sequence_id": 0, "start_index": 115, "end_index": 117, "count": 3}
    )
    for local_index, (embedding, position, timestamp) in enumerate(
        [
            ([0.0, 0.0], [0.0, 0.0, 0.0], 1_672_000_010),
            ([1.0, 0.0], [1.0, 0.0, 0.0], 1_672_000_020),
            ([9.0, 0.0], [2.0, 0.0, 0.0], 1_672_000_030),
        ]
    ):
        item = MemoryItem(
            caption=f"caption {local_index}",
            time=timestamp,
            position=position,
            theta=0.0,
        )
        item.entry_id = 115 + local_index
        item.source_file_start = f"{timestamp}.pkl"
        item.source_file_end = f"{timestamp + 3}.pkl"
        memory.insert(item, text_embedding=embedding)
    return memory


class LocalVectorMemoryTraceTest(unittest.TestCase):
    def test_text_trace_has_stable_ids_full_ranking_and_returned_context(self):
        memory = make_memory()

        returned = memory.search_by_text("door")
        trace = memory.get_retrieval_trace()

        self.assertEqual([row["entry_id"] for row in trace[0]["selected"]], [115, 116])
        self.assertEqual([row["entry_id"] for row in trace[0]["ranking"]], [115, 116, 117])
        self.assertEqual(trace[0]["score_name"], "embedding_l2_distance")
        self.assertEqual(trace[0]["returned_context"], returned)
        self.assertIn("caption 0", returned)
        json.dumps(trace)

    def test_position_and_time_traces_record_parsed_queries_and_scores(self):
        memory = make_memory()

        memory.search_by_position((1.9, 0.0, 0.0))
        requested_clock = __import__("time").strftime(
            "%H:%M:%S", __import__("time").localtime(1_672_000_021)
        )
        memory.search_by_time(requested_clock)
        trace = memory.get_retrieval_trace()

        self.assertEqual(trace[0]["selected"][0]["entry_id"], 117)
        self.assertAlmostEqual(trace[0]["selected"][0]["score"], 0.1, places=5)
        self.assertEqual(trace[1]["selected"][0]["entry_id"], 116)
        self.assertEqual(trace[1]["selected"][0]["score"], 1.0)
        self.assertEqual(trace[1]["score_unit"], "seconds")
        json.dumps(trace)

    def test_trace_getters_return_copies(self):
        memory = make_memory()
        memory.search_by_text("hall")

        external = memory.get_retrieval_trace()
        external[0]["selected"].clear()
        metadata = memory.get_candidate_pool_metadata()
        metadata["count"] = 0

        self.assertEqual(len(memory.get_retrieval_trace()[0]["selected"]), 2)
        self.assertEqual(memory.get_candidate_pool_metadata()["count"], 3)

    def test_question_budget_blocks_duplicate_dense_evidence_and_resets(self):
        memory = make_memory(text_episode_mode="question")
        memory.begin_retrieval_episode()

        memory.search_by_text("door")
        self.assertFalse(memory.text_retrieval_available())
        exhausted = memory.search_by_text("hall")
        traces = memory.get_retrieval_trace()

        self.assertEqual(
            [row["entry_id"] for row in traces[0]["selected"]],
            [115, 116],
        )
        self.assertEqual(traces[1]["selected"], [])
        self.assertTrue(traces[1]["budget_exhausted"])
        self.assertIn("evidence budget is exhausted", exhausted)

        memory.begin_retrieval_episode()
        self.assertTrue(memory.text_retrieval_available())
        memory.search_by_text("hall")
        retry = memory.get_retrieval_trace()[2]
        self.assertEqual(retry["text_episode_id"], 2)
        self.assertEqual(
            [row["entry_id"] for row in retry["selected"]],
            [117, 116],
        )

    def test_unified_top1_masks_ids_across_text_position_and_time(self):
        memory = make_memory(unified_evidence_ledger=True)
        memory.begin_retrieval_episode()

        memory.search_by_position((2.0, 0.0, 0.0))
        memory.search_by_text("door")
        requested_clock = __import__("time").strftime(
            "%H:%M:%S", __import__("time").localtime(1_672_000_020)
        )
        memory.search_by_time(requested_clock)
        traces = memory.get_retrieval_trace()

        self.assertEqual(
            [trace["selected"][0]["entry_id"] for trace in traces],
            [117, 115, 116],
        )
        self.assertTrue(all(trace["returned_count"] == 1 for trace in traces))
        self.assertEqual(traces[1]["global_selected_entry_ids_before"], [117])
        self.assertEqual(
            traces[2]["global_selected_entry_ids_after"],
            [117, 115, 116],
        )
        self.assertEqual(traces[2]["evidence_state_version"], 3)
        self.assertEqual(
            [source["tool"] for source in traces[2]["prior_evidence_sources"]],
            ["retrieve_from_position", "retrieve_from_text"],
        )
        self.assertTrue(traces[2]["budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
