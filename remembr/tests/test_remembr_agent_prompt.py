import json
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from remembr.agents.remembr_agent import ReMEmbRAgent
from remembr.tools.retrieval_control import RetrievalCallGate, tool_call_signature


class ControllerPromptTest(unittest.TestCase):
    def test_embedded_json_is_not_parsed_as_template_variables(self):
        policy = '{"context_reasoning": "inspect evidence", "tool": "retrieve"}'
        ledger = 'prior call: {"x": "green exit sign"}'

        prompt = ReMEmbRAgent._controller_chat_prompt(policy, ledger)
        rendered = prompt.invoke({"question": "Where is the exit?", "chat_history": []})

        self.assertEqual(prompt.input_variables, ["chat_history", "question"])
        self.assertEqual(rendered.messages[0].content, policy)
        self.assertEqual(rendered.messages[1].content, ledger)

    def test_duplicate_is_reprompted_once_before_forcing_reader(self):
        agent = ReMEmbRAgent.__new__(ReMEmbRAgent)
        agent.answer_attempt_id = "attempt_1"
        agent.controller_turn_id = 2
        agent.retrieval_gate = RetrievalCallGate(5, duplicate_replan_limit=2)
        _, signature = tool_call_signature(
            "retrieve_from_text", {"x": "green exit sign"}
        )
        agent.retrieval_gate.commit(signature, [115])
        agent.retrieval_control_trace = []
        agent.force_reader = False
        agent.tools_by_name = {"retrieve_from_text": object()}
        agent.memory = SimpleNamespace()

        def duplicate_state(call_id):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "retrieve_from_text",
                                "args": {"x": " Green  EXIT sign "},
                                "id": call_id,
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }

        first = agent.call_tool(duplicate_state("duplicate-1"))
        first_payload = json.loads(first["messages"][0].content)
        self.assertEqual(first_payload["type"], "invalid_retrieval_request")
        self.assertTrue(agent.retrieval_control_trace[0]["duplicate_reprompted"])
        self.assertFalse(agent.force_reader)
        self.assertEqual(agent.retrieval_gate.round_count, 1)

        agent.controller_turn_id = 3
        agent.call_tool(duplicate_state("duplicate-2"))
        self.assertFalse(agent.retrieval_control_trace[1]["duplicate_reprompted"])
        self.assertEqual(
            agent.retrieval_control_trace[1]["forced_stop_reason"],
            "duplicate_replan_limit",
        )
        self.assertTrue(agent.force_reader)
        self.assertEqual(agent.retrieval_gate.round_count, 1)


if __name__ == "__main__":
    unittest.main()
