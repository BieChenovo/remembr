import unittest

from remembr.agents.remembr_agent import ReMEmbRAgent


class ControllerPromptTest(unittest.TestCase):
    def test_embedded_json_is_not_parsed_as_template_variables(self):
        policy = '{"context_reasoning": "inspect evidence", "tool": "retrieve"}'
        ledger = 'prior call: {"x": "green exit sign"}'

        prompt = ReMEmbRAgent._controller_chat_prompt(policy, ledger)
        rendered = prompt.invoke({"question": "Where is the exit?", "chat_history": []})

        self.assertEqual(prompt.input_variables, ["chat_history", "question"])
        self.assertEqual(rendered.messages[0].content, policy)
        self.assertEqual(rendered.messages[1].content, ledger)


if __name__ == "__main__":
    unittest.main()
