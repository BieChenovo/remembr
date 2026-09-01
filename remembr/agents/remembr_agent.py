from typing import Annotated, Any, Iterator, List, Literal, Optional, Sequence, TypedDict
import ast
import copy
import json
import traceback
import sys, re

import requests

# from langchain_openai import OpenAIEmbeddings

from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field


import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper
from remembr.tools.retrieval_control import (
    RETRIEVAL_TOOL_NAMES,
    RetrievalCallGate,
    merge_controller_trace,
    qrag_state_components,
    raw_tool_query,
    selected_entry_ids,
    tool_call_signature,
)

from remembr.memory.memory import Memory

from remembr.agents.agent import Agent, AgentOutput


class ThinkAwareChatOllama(ChatOllama):
    """Backport Ollama's top-level ``think`` option to old LangChain.

    ``langchain-community==0.2`` predates this Ollama request field. Passing it
    as a normal invocation kwarg incorrectly nests it under ``options``; modern
    Ollama then keeps Qwen3's reasoning in ``message.thinking`` while LangChain
    sees an empty ``message.content``. This adapter places the field where the
    Ollama API expects it.
    """

    think: Optional[bool] = None

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        if self.stop is not None:
            stop = self.stop

        params = self._default_params
        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{key: value for key, value in kwargs.items() if key not in self._default_params},
            }

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }
        if self.think is not None:
            request_payload["think"] = self.think

        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            auth=self.auth,
            json=request_payload,
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}. "
                f"Details: {response.text}"
            )
        return response.iter_lines(decode_unicode=True)



### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    # print(state)
    return state


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return ast.literal_eval(parsed)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func, max_attempts=3):
    last_error = None
    for _ in range(max_attempts):
        try:
            ret = func(state)
            return ret
        except Exception as e:
            last_error = e
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
    raise RuntimeError(
        f"{func} failed after {max_attempts} attempts"
    ) from last_error

class ReMEmbRAgent(Agent):

    def __init__(
        self,
        llm_type='gpt-4o',
        num_ctx=8192,
        temperature=0,
        num_predict=2048,
        disable_thinking=False,
        max_retrieval_rounds=5,
        duplicate_replan_limit=2,
    ):

        if int(max_retrieval_rounds) < 1:
            raise ValueError("max_retrieval_rounds must be positive")
        if int(duplicate_replan_limit) < 1:
            raise ValueError("duplicate_replan_limit must be positive")

        # Wrapper that handles everything
        llm = self.llm_selector(
            llm_type,
            temperature,
            num_ctx,
            num_predict,
            disable_thinking,
        )
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature
        self.num_predict = num_predict
        self.disable_thinking = disable_thinking
        self.max_retrieval_rounds = int(max_retrieval_rounds)
        self.duplicate_replan_limit = int(duplicate_replan_limit)

        self.chat = chat
        self.llm_type = llm_type
        ### Load vectorstore
        # self.update_for_instance() # ref_time is None this time
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+'prompts/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.answer_attempt_count = 0
        self.answer_attempt_id = None
        self.controller_turn_id = 0
        self.retrieval_gate = RetrievalCallGate(
            self.max_retrieval_rounds,
            self.duplicate_replan_limit,
        )
        self.retrieval_control_trace = []
        self.force_reader = False

        self.chat_history = ChatMessageHistory()


    def llm_selector(
        self,
        llm_type,
        temperature,
        num_ctx,
        num_predict,
        disable_thinking,
    ):
        llm = None
        # langchain-community 0.2 hard-codes localhost:11434 and does not read
        # OLLAMA_HOST itself. Respect the standard Ollama environment variable
        # so parallel workers can be pinned to independent GPU-local servers.
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host and "://" not in ollama_host:
            ollama_host = f"http://{ollama_host}"
        ollama_connection = {"base_url": ollama_host} if ollama_host else {}
        # Support for LLM Gateway
        if 'gpt-4' in llm_type:
            # TODO: ADD OpenAI here
            pass

        # Support for NIMs
        elif 'nim/' in llm_type:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            llm_name = llm_type[4:]
            llm = ChatNVIDIA(model=llm_name)

        # Support for Ollama functions
        elif llm_type == 'command-r':
            llm = ChatOllama(
                model=llm_type,
                temperature=temperature,
                num_ctx=num_ctx,
                num_predict=num_predict,
                **ollama_connection,
            )
        else:
            llm_class = (
                ThinkAwareChatOllama
                if llm_type.lower().startswith("qwen3")
                else ChatOllama
            )
            llm = llm_class(
                model=llm_type,
                format="json",
                temperature=temperature,
                num_ctx=num_ctx,
                num_predict=num_predict,
                **ollama_connection,
                **(
                    {"think": False}
                    if disable_thinking and llm_class is ThinkAwareChatOllama
                    else {}
                ),
            )

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm

    def generation_directive(self, question):
        """Disable Qwen3 reasoning tokens when requested by the evaluator.

        The installed LangChain Ollama adapter predates Ollama's top-level
        ``think`` request field. Qwen3 also supports the equivalent per-turn
        ``/no_think`` directive, which keeps structured JSON in the visible
        response instead of spending the output budget on hidden reasoning.
        """
        if self.disable_thinking and self.llm_type.lower().startswith('qwen3'):
            return question + "\n/no_think"
        return question


    def set_memory(self, memory: Memory):
        self.memory = memory
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.answer_attempt_count = 0
        self.answer_attempt_id = None
        self.controller_turn_id = 0
        self.retrieval_gate = RetrievalCallGate(
            self.max_retrieval_rounds,
            self.duplicate_replan_limit,
        )
        self.retrieval_control_trace = []
        self.force_reader = False
        self.chat_history = ChatMessageHistory()
        self.create_tools(memory)
        self.build_graph()

    def get_retrieval_trace(self):
        getter = getattr(self.memory, "get_retrieval_trace", None)
        memory_trace = getter() if getter is not None else []
        return merge_controller_trace(memory_trace, self.retrieval_control_trace)

    def get_candidate_pool_metadata(self):
        getter = getattr(self.memory, "get_candidate_pool_metadata", None)
        return getter() if getter is not None else {}



    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                The query must be an (x,y,z) array with floating point values \
                                Based on the question and your context, decide what position to search for in the database. \
                                This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                                The query will then search your memories for you.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tools_by_name = {tool.name: tool for tool in self.tool_list}
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    def _local_model_history(self, messages):
        """Serialize tool provenance for old ChatOllama adapters.

        LangChain 0.2's community Ollama adapter cannot encode ToolMessage, but
        converting it to AIMessage falsely presents memory output as something
        the controller said.  System records retain the tool, arguments, call
        ID, selected IDs, and result while keeping the provenance explicit.
        """

        if ('gpt-4' in self.llm_type) or ('nim' in self.llm_type):
            return list(messages)

        calls_by_id = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    calls_by_id[call.get("id")] = call

        converted = []
        for message in messages:
            if isinstance(message, ToolMessage):
                call = calls_by_id.get(message.tool_call_id, {})
                payload = {
                    "type": "retrieval_tool_result",
                    "tool_call_id": message.tool_call_id,
                    "tool": getattr(message, "name", None) or call.get("name"),
                    "arguments": call.get("args"),
                    "result": message.content,
                }
                converted.append(
                    SystemMessage(
                        content="RETRIEVAL_TOOL_RESULT\n"
                        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    )
                )
            elif isinstance(message, AIMessage) and message.tool_calls:
                if message.content:
                    converted.append(AIMessage(content=message.content))
                converted.append(
                    SystemMessage(
                        content="CONTROLLER_TOOL_REQUEST\n"
                        + json.dumps(
                            [
                                {
                                    "tool_call_id": call.get("id"),
                                    "tool": call.get("name"),
                                    "arguments": call.get("args"),
                                }
                                for call in message.tool_calls
                            ],
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                )
            else:
                converted.append(message)
        return converted

    @staticmethod
    def _tool_result_payload(
        tool_call_id,
        tool_name,
        raw_query,
        selected_ids,
        result,
        duplicate_blocked=False,
    ):
        return json.dumps(
            {
                "type": "retrieval_tool_result",
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "query": raw_query,
                "selected_ids": list(selected_ids),
                "duplicate_blocked": bool(duplicate_blocked),
                "result": str(result),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _invalid_retrieval_payload(
        tool_call_id,
        tool_name,
        raw_query,
        executed_retrieval_rounds,
        duplicate_replan_count,
        duplicate_replan_limit,
        forced_stop,
    ):
        return json.dumps(
            {
                "type": "invalid_retrieval_request",
                "reason": "duplicate_query",
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "query": raw_query,
                "retrieval_executed": False,
                "executed_retrieval_rounds": executed_retrieval_rounds,
                "duplicate_replan_count": duplicate_replan_count,
                "duplicate_replan_limit": duplicate_replan_limit,
                "forced_stop": bool(forced_stop),
                "instruction": (
                    "Use the visible result to answer, switch modality, or "
                    "formulate a semantically different query. Do not make an "
                    "unsupported cosmetic change merely to bypass deduplication."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _retrieval_kind(tool_name, memory_record):
        if tool_name in {"retrieve_from_time", "retrieve_from_position"}:
            return "non_qrag"
        method = str(memory_record.get("retrieval_method", ""))
        return "qrag" if method.startswith("qrag_") else "dense"

    ### Nodes

    @staticmethod
    def _controller_chat_prompt(policy, controller_ledger):
        """Build a controller prompt without templating embedded JSON."""

        return ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=policy),
                SystemMessage(content=controller_ledger),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

    def _controller_ledger(self):
        """Describe the attempt state and exact signatures that are disabled."""

        return "RETRIEVAL_CONTROLLER_LEDGER\n" + json.dumps(
            {
                "type": "retrieval_controller_ledger",
                "executed_retrieval_rounds": self.retrieval_gate.round_count,
                "max_executed_retrieval_rounds": self.retrieval_gate.max_rounds,
                "disabled_signatures": sorted(
                    self.retrieval_gate.executed_signatures
                ),
                "disabled_requests": [
                    {
                        "tool": event.get("tool"),
                        "normalized_query": event.get("normalized_query"),
                        "signature": event.get("tool_signature"),
                    }
                    for event in self.retrieval_control_trace
                    if event.get("answer_attempt_id") == self.answer_attempt_id
                    and event.get("retrieval_executed") is True
                ],
                "visible_result_ids": list(
                    self.retrieval_gate.visible_result_ids
                ),
                "consecutive_duplicate_replans": (
                    self.retrieval_gate.consecutive_duplicate_replans
                ),
                "duplicate_replan_limit": (
                    self.retrieval_gate.duplicate_replan_limit
                ),
                "instruction": (
                    "Do not repeat a disabled signature. Answer, switch modality, "
                    "or formulate a semantically different evidence query. Do not "
                    "make unsupported cosmetic changes to time, coordinates, or text."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]
        self.controller_turn_id += 1
        self.agent_call_count = self.controller_turn_id

        can_retrieve = (
            not self.force_reader
            and self.retrieval_gate.can_retrieve
        )
        model = self.chat
        if can_retrieve:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt

        # Controller policy is a system instruction, not an assistant utterance.
        # Keep the policy and controller ledger as concrete messages.  Treating
        # either string as a prompt template makes the JSON examples (and later
        # JSON-encoded tool arguments) look like undeclared template variables.
        agent_prompt = self._controller_chat_prompt(
            prompt,
            self._controller_ledger(),
        )
        model = agent_prompt | model
        question = self.generation_directive(
            f"The question is: {messages[0].content}"
        )
        response = model.invoke(
            {
                "question": question,
                "chat_history": self._local_model_history(messages),
            }
        )

        # FunctionsWrapper rejects batches, but keep this invariant explicit at
        # the orchestration boundary as well for non-Ollama model adapters.
        if len(response.tool_calls) > 1:
            raise ValueError(
                "A controller turn produced multiple retrieval calls; none executed"
            )
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call["name"] not in RETRIEVAL_TOOL_NAMES:
                raise ValueError(f"Unsupported retrieval tool: {tool_call['name']}")

        return {"messages": [response]}

    def call_tool(self, state):
        """Execute one retrieval call and attach controller-level provenance."""

        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or len(message.tool_calls) != 1:
            raise ValueError("The action node requires exactly one retrieval call")

        tool_call = message.tool_calls[0]
        tool_name = tool_call.get("name")
        if tool_name not in self.tools_by_name:
            raise ValueError(f"Unknown retrieval tool: {tool_name}")
        arguments = copy.deepcopy(tool_call.get("args") or {})
        raw_query = copy.deepcopy(raw_tool_query(arguments))
        normalized_query, signature = tool_call_signature(tool_name, arguments)
        tool_call_id = tool_call.get("id")
        turn_id = self.controller_turn_id
        evidence_ledger_getter = getattr(self.memory, "get_evidence_ledger", None)
        evidence_ledger = (
            evidence_ledger_getter() if evidence_ledger_getter is not None else []
        )
        global_entry_ids = [
            source.get("entry_id")
            for source in evidence_ledger
            if isinstance(source, dict) and source.get("entry_id") is not None
        ]
        evidence_state_version = int(
            getattr(self.memory, "_evidence_state_version", 0)
        )
        event = {
            "answer_attempt_id": self.answer_attempt_id,
            "controller_turn_id": turn_id,
            "tool_batch_id": f"{self.answer_attempt_id}:turn_{turn_id}",
            "tool_batch_size": 1,
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "raw_arguments": arguments,
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "tool_signature": signature,
            "duplicate_blocked": False,
            "retrieval_executed": False,
            "duplicate_reprompted": False,
            "duplicate_replan_count": (
                self.retrieval_gate.consecutive_duplicate_replans
            ),
            "duplicate_replan_limit": self.retrieval_gate.duplicate_replan_limit,
            "executed_retrieval_rounds": self.retrieval_gate.round_count,
            "forced_stop_reason": None,
            "evidence_state_version_before": evidence_state_version,
            "evidence_state_version": evidence_state_version,
            "global_selected_entry_ids_before": list(global_entry_ids),
            "global_selected_entry_ids_after": list(global_entry_ids),
            "prior_evidence_sources": evidence_ledger,
            "prior_result_ids_visible_to_controller": list(
                self.retrieval_gate.visible_result_ids
            ),
            "selected_ids": [],
            "qrag_state_components": [],
        }

        if self.retrieval_gate.is_duplicate(signature):
            duplicate_count, reached_limit = self.retrieval_gate.record_duplicate()
            reason = self._invalid_retrieval_payload(
                tool_call_id,
                tool_name,
                raw_query,
                self.retrieval_gate.round_count,
                duplicate_count,
                self.retrieval_gate.duplicate_replan_limit,
                reached_limit,
            )
            event.update(
                {
                    "duplicate_blocked": True,
                    "duplicate_reprompted": not reached_limit,
                    "duplicate_replan_count": duplicate_count,
                    "duplicate_replan_limit": (
                        self.retrieval_gate.duplicate_replan_limit
                    ),
                    "executed_retrieval_rounds": self.retrieval_gate.round_count,
                    "forced_stop_reason": (
                        "duplicate_replan_limit" if reached_limit else None
                    ),
                    "blocked_reason": reason,
                    "retrieval_kind": (
                        "non_qrag"
                        if tool_name in {
                            "retrieve_from_time",
                            "retrieve_from_position",
                        }
                        else "qrag"
                        if self.memory.__class__.__name__ == "QragLocalMemory"
                        else "dense"
                    ),
                }
            )
            self.retrieval_control_trace.append(event)
            # Let the controller use the structured correction to answer,
            # switch modality, or formulate a genuinely different query.  A
            # finite consecutive-error limit prevents deterministic loops.
            self.force_reader = reached_limit
            return {
                "messages": [
                    ToolMessage(
                        content=reason,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                ]
            }

        if not self.retrieval_gate.can_retrieve:
            raise ValueError("Retrieval round limit reached before tool execution")

        getter = getattr(self.memory, "get_retrieval_trace", None)
        before_trace = getter() if getter is not None else []
        result = self.tools_by_name[tool_name].invoke(arguments)
        after_trace = getter() if getter is not None else []
        memory_trace_index = (
            len(before_trace) if len(after_trace) > len(before_trace) else None
        )
        memory_record = (
            after_trace[memory_trace_index]
            if memory_trace_index is not None
            else {}
        )
        returned_ids = selected_entry_ids(memory_record)

        retrieval_round_id = self.retrieval_gate.commit(signature, returned_ids)
        event.update(
            {
                "memory_trace_index": memory_trace_index,
                "retrieval_round_id": retrieval_round_id,
                "retrieval_executed": True,
                "duplicate_reprompted": False,
                "duplicate_replan_count": 0,
                "executed_retrieval_rounds": self.retrieval_gate.round_count,
                "selected_ids": returned_ids,
                "qrag_state_components": qrag_state_components(memory_record),
                "retrieval_kind": self._retrieval_kind(
                    tool_name,
                    memory_record,
                ),
            }
        )
        for key in (
            "evidence_state_version",
            "evidence_state_version_before",
            "global_selected_entry_ids_before",
            "global_selected_entry_ids_after",
            "prior_evidence_sources",
        ):
            if key in memory_record:
                event[key] = copy.deepcopy(memory_record[key])
        self.retrieval_control_trace.append(event)
        if not self.retrieval_gate.can_retrieve:
            self.force_reader = True
            event["forced_stop_reason"] = "retrieval_round_limit"

        return {
            "messages": [
                ToolMessage(
                    content=self._tool_result_payload(
                        tool_call_id,
                        tool_name,
                        raw_query,
                        returned_ids,
                        result,
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            ]
        }


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = self.generation_directive(
            messages[0].content + "\n Please responsed in the desired format."
        )
        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})


        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("human", "What do you do?"),
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                # ("ai", filled_prompt.text),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke(
            {
                "question": question,
                "chat_history": self._local_model_history(messages[1:]),
            }
        )

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = ast.literal_eval(response)
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")
                
            if type(parsed['position']) == str:
                parsed['position'] = ast.literal_eval(parsed['position'])
            
            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}



    def build_graph(self):

        from langgraph.graph import END, StateGraph

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        workflow.add_node("action", self.call_tool)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )


        workflow.add_edge('action', 'agent')

        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str):

        # ``answer_squad_question`` may retry the same question after a parse or
        # evaluation failure.  Each retry is an independent retrieval episode:
        # keep the trace history for audit, but do not inherit consumed Q-RAG
        # budget, selected-ID masks, or stale controller tool requests.
        begin_episode = getattr(self.memory, "begin_retrieval_episode", None)
        if begin_episode is not None:
            begin_episode()
        self.answer_attempt_count += 1
        self.answer_attempt_id = f"attempt_{self.answer_attempt_count}"
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.controller_turn_id = 0
        self.retrieval_gate.reset()
        self.force_reader = False

        inputs = { "messages": [
                                (("user", question)),
            ]
        }

        out = self.graph.invoke(inputs)
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = ast.literal_eval(response)
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)


        return response

if __name__ == "__main__":

    from memory.milvus_memory import MilvusMemory

    # llm_name = 
    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs (such as 'codestral')
    memory = MilvusMemory("test", db_ip='127.0.0.1')

    llm_name = 'gpt-4o' 
    agent = ReMEmbRAgent(llm_type=llm_name)

    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")
