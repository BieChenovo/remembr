import unittest

from remembr.tools.retrieval_control import (
    ControllerToolChoiceError,
    RetrievalCallGate,
    ensure_single_tool_choice,
    merge_controller_trace,
    tool_call_signature,
)


class RetrievalControlTest(unittest.TestCase):
    def test_parallel_or_mixed_controller_choices_are_rejected(self):
        with self.assertRaises(ControllerToolChoiceError):
            ensure_single_tool_choice(
                [
                    {"tool": "retrieve_from_text", "tool_input": {"x": "door"}},
                    {"tool": "retrieve_from_time", "tool_input": {"x": "07:55:32"}},
                ]
            )

    def test_text_signature_normalizes_case_and_whitespace(self):
        first = tool_call_signature(
            "retrieve_from_text",
            {"x": "  Green   EXIT Sign "},
        )
        second = tool_call_signature(
            "retrieve_from_text",
            {"x": "green exit sign"},
        )
        self.assertEqual(first, second)

    def test_time_and_position_signatures_normalize_equivalent_queries(self):
        self.assertEqual(
            tool_call_signature("retrieve_from_time", {"x": "7:05:02"}),
            tool_call_signature("retrieve_from_time", {"x": "07:05:02"}),
        )
        self.assertEqual(
            tool_call_signature("retrieve_from_position", {"x": "(1, 2, 3)"}),
            tool_call_signature("retrieve_from_position", {"x": [1, 2, 3]}),
        )

    def test_duplicate_signature_is_detected_without_consuming_a_round(self):
        gate = RetrievalCallGate(max_rounds=2)
        _, signature = tool_call_signature(
            "retrieve_from_time",
            {"x": "07:55:32"},
        )
        gate.commit(signature, [115, 116])

        self.assertTrue(gate.is_duplicate(signature))
        self.assertEqual(gate.round_count, 1)
        self.assertEqual(gate.visible_result_ids, [115, 116])
        with self.assertRaises(ValueError):
            gate.commit(signature, [115, 116])
        self.assertEqual(gate.round_count, 1)

    def test_duplicate_replans_are_bounded_and_reset_by_success(self):
        gate = RetrievalCallGate(max_rounds=3, duplicate_replan_limit=2)
        _, first = tool_call_signature("retrieve_from_text", {"x": "door"})
        _, second = tool_call_signature("retrieve_from_text", {"x": "hall"})
        gate.commit(first, [115])

        self.assertEqual(gate.record_duplicate(), (1, False))
        self.assertEqual(gate.round_count, 1)
        gate.commit(second, [116])
        self.assertEqual(gate.consecutive_duplicate_replans, 0)
        self.assertEqual(gate.record_duplicate(), (1, False))
        self.assertEqual(gate.record_duplicate(), (2, True))
        self.assertEqual(gate.round_count, 2)

    def test_trace_merge_proves_second_turn_saw_first_result(self):
        memory_trace = [
            {"call_index": 1, "selected": [{"entry_id": 115}]},
            {"call_index": 2, "selected": [{"entry_id": 117}]},
        ]
        events = [
            {
                "memory_trace_index": 0,
                "controller_turn_id": 1,
                "prior_result_ids_visible_to_controller": [],
                "selected_ids": [115],
            },
            {
                "memory_trace_index": 1,
                "controller_turn_id": 2,
                "prior_result_ids_visible_to_controller": [115],
                "selected_ids": [117],
            },
        ]

        merged = merge_controller_trace(memory_trace, events)

        self.assertEqual(merged[0]["selected_ids"], [115])
        self.assertEqual(merged[1]["controller_turn_id"], 2)
        self.assertEqual(
            merged[1]["prior_result_ids_visible_to_controller"],
            [115],
        )


if __name__ == "__main__":
    unittest.main()
