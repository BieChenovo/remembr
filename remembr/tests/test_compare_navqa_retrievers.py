import unittest

from remembr.scripts.compare_navqa_retrievers import (
    portable_config,
    retrieval_calls_for_scoring,
)


class RetrievalCallsForScoringTest(unittest.TestCase):
    def test_uses_only_final_successful_attempt(self):
        failed_call = {"tool": "retrieve_from_text", "selected": [{"entry_id": 1}]}
        succeeded_call = {"tool": "retrieve_from_time", "selected": [{"entry_id": 2}]}
        response = {
            "retrieval_trace": [failed_call, succeeded_call],
            "retrieval_attempts": [
                {"status": "failed", "calls": [failed_call]},
                {"status": "succeeded", "calls": [succeeded_call]},
            ],
        }

        calls, available = retrieval_calls_for_scoring(response)

        self.assertTrue(available)
        self.assertEqual(calls, [succeeded_call])

    def test_successful_zero_call_attempt_is_traceable(self):
        response = {
            "retrieval_trace": [{"tool": "retrieve_from_text"}],
            "retrieval_attempts": [{"status": "succeeded", "calls": []}],
        }

        calls, available = retrieval_calls_for_scoring(response)

        self.assertTrue(available)
        self.assertEqual(calls, [])

    def test_all_failed_retries_use_only_final_attempt(self):
        first = {"tool": "retrieve_from_text", "selected": [{"entry_id": 1}]}
        final = {"tool": "retrieve_from_text", "selected": [{"entry_id": 2}]}
        response = {
            "retrieval_trace": [first, final],
            "retrieval_attempts": [
                {"status": "failed", "calls": [first]},
                {"status": "failed", "calls": [final]},
            ],
        }

        calls, available = retrieval_calls_for_scoring(response)

        self.assertTrue(available)
        self.assertEqual(calls, [final])

    def test_old_result_without_trace_is_unverifiable(self):
        calls, available = retrieval_calls_for_scoring({})

        self.assertFalse(available)
        self.assertEqual(calls, [])

    def test_published_configs_do_not_expose_machine_paths(self):
        config = {
            "questions": "/machine/user/projects/remembr/artifacts/questions/0.json",
            "nested": ["/scratch/projects/remembr/artifacts/cache"],
        }

        self.assertEqual(
            portable_config(config),
            {
                "questions": "remembr/artifacts/questions/0.json",
                "nested": ["remembr/artifacts/cache"],
            },
        )


if __name__ == "__main__":
    unittest.main()
