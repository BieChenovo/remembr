"""Small in-process vector memory for NaVQA evaluation.

The evaluation sequences contain at most hundreds of caption entries, so a
NumPy scan is simpler and more reproducible than requiring a Milvus service.
Caption embeddings generated offline are reused; only text queries are encoded.
"""

import datetime
import copy
import time
from time import localtime, strftime

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from remembr.memory.memory import Memory, MemoryItem


class LocalVectorMemory(Memory):
    _embedder_cache = {}

    def __init__(
        self,
        embedding_model,
        time_offset,
        text_k=5,
        numeric_k=4,
        text_episode_mode="per_call",
        question_text_evidence_budget=None,
        unified_evidence_ledger=False,
        question_evidence_budget=None,
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
        if question_evidence_budget is not None and int(question_evidence_budget) < 1:
            raise ValueError("Question evidence budget must be positive")
        if embedding_model not in self._embedder_cache:
            self._embedder_cache[embedding_model] = HuggingFaceEmbeddings(
                model_name=embedding_model
            )
        self.embedder = self._embedder_cache[embedding_model]
        self.time_offset = time_offset
        self.text_k = text_k
        self.numeric_k = numeric_k
        self.text_episode_mode = text_episode_mode
        self.question_text_evidence_budget = int(
            question_text_evidence_budget
            if question_text_evidence_budget is not None
            else text_k
        )
        self.unified_evidence_ledger = bool(unified_evidence_ledger)
        self.question_evidence_budget = int(
            question_evidence_budget
            if question_evidence_budget is not None
            else self.question_text_evidence_budget
        )
        self.reset()

    def reset(self):
        self.items = []
        self.text_embeddings = []
        self.working_memory = []
        self.retrieval_trace = []
        self.candidate_pool_metadata = {}
        self._text_episode_id = 0
        self._text_episode_selected_indices = []
        self._evidence_episode_id = 0
        self._evidence_selected_indices = []
        self._evidence_sources = []
        self._evidence_state_version = 0

    def begin_retrieval_episode(self):
        """Reset question-level budgets and masks for a new answer attempt."""

        self._text_episode_id += 1
        self._text_episode_selected_indices = []
        self._evidence_episode_id += 1
        self._evidence_selected_indices = []
        self._evidence_sources = []
        self._evidence_state_version = 0

    def text_retrieval_available(self):
        if not self.items:
            return False
        if getattr(self, "unified_evidence_ledger", False):
            total_budget = int(getattr(self, "question_evidence_budget", self.text_k))
            return (
                len(self._evidence_selected_indices) < total_budget
                and len(self._evidence_selected_indices) < len(self.items)
            )
        if getattr(self, "text_episode_mode", "per_call") != "question":
            return True
        total_budget = int(
            getattr(self, "question_text_evidence_budget", self.text_k)
        )
        return (
            len(self._text_episode_selected_indices) < total_budget
            and len(self._text_episode_selected_indices) < len(self.items)
        )

    def _text_episode_policy(self, k):
        if getattr(self, "text_episode_mode", "per_call") != "question":
            return set(), int(k), None
        prior = set(self._text_episode_selected_indices)
        total_budget = int(
            getattr(self, "question_text_evidence_budget", self.text_k)
        )
        remaining = max(total_budget - len(prior), 0)
        return prior, min(int(k), remaining), remaining

    def _commit_text_episode_selection(self, indices):
        if getattr(self, "text_episode_mode", "per_call") != "question":
            return
        seen = set(self._text_episode_selected_indices)
        for index in indices:
            index = int(index)
            if index not in seen:
                self._text_episode_selected_indices.append(index)
                seen.add(index)

    def _selection_policy(self, tool, k):
        """Return the attempt-scoped mask and effective per-call budget."""

        if getattr(self, "unified_evidence_ledger", False):
            prior_order = list(self._evidence_selected_indices)
            total_budget = int(getattr(self, "question_evidence_budget", self.text_k))
            remaining = max(total_budget - len(prior_order), 0)
            # Unified v4 retrieval exposes one new memory per controller turn.
            effective_k = min(int(k), 1, remaining)
            return set(prior_order), prior_order, effective_k, remaining, "unified"

        if tool == "retrieve_from_text":
            prior, effective_k, remaining = self._text_episode_policy(k)
            prior_order = (
                list(self._text_episode_selected_indices)
                if getattr(self, "text_episode_mode", "per_call") == "question"
                else []
            )
            return prior, prior_order, effective_k, remaining, "text"

        return set(), [], int(k), None, None

    def _commit_evidence_selection(self, indices, *, tool, query):
        """Commit unique memories to the shared cross-modal v4 ledger."""

        if not getattr(self, "unified_evidence_ledger", False):
            if tool == "retrieve_from_text":
                self._commit_text_episode_selection(indices)
            return

        seen = set(self._evidence_selected_indices)
        for index in indices:
            index = int(index)
            if index in seen:
                continue
            item = self.items[index]
            self._evidence_selected_indices.append(index)
            self._evidence_sources.append(
                {
                    "entry_id": getattr(item, "entry_id", None),
                    "local_index": index,
                    "tool": tool,
                    "query": copy.deepcopy(query),
                    "time": float(item.time),
                    "timestamp": float(item.time),
                    "position": np.asarray(item.position, dtype=float).tolist(),
                }
            )
            self._evidence_state_version += 1
            seen.add(index)

    def get_evidence_ledger(self):
        """Return an audit-safe copy of the shared evidence provenance."""

        return copy.deepcopy(self._evidence_sources)

    @staticmethod
    def empty_text_result_context(budget_exhausted):
        if budget_exhausted:
            return (
                "No additional text memories were retrieved because the "
                "question-level evidence budget is exhausted."
            )
        return "No additional text memories were available."

    def set_candidate_pool_metadata(self, metadata):
        """Attach evaluator-side bounds to the next retrieval trace.

        The local item index is relative to one question's candidate pool, while
        ``entry_id`` is the stable index in the full sequence caption file.  We
        keep both so audit reports cannot accidentally confuse the two.
        """
        self.candidate_pool_metadata = copy.deepcopy(metadata)

    def get_candidate_pool_metadata(self):
        return copy.deepcopy(self.candidate_pool_metadata)

    def get_retrieval_trace(self):
        return copy.deepcopy(self.retrieval_trace)

    def insert(self, item: MemoryItem, text_embedding=None):
        self.items.append(item)
        if text_embedding is None:
            text_embedding = self.embedder.embed_query(item.caption)
        self.text_embeddings.append(np.asarray(text_embedding, dtype=np.float32))

    def get_working_memory(self):
        return self.working_memory

    @staticmethod
    def _item_record(item, local_index):
        return {
            "entry_id": getattr(item, "entry_id", None),
            "local_index": int(local_index),
            "caption": item.caption,
            "time": float(item.time),
            "position": np.asarray(item.position, dtype=float).tolist(),
            "theta": float(item.theta),
            "source_file_start": getattr(item, "source_file_start", None),
            "source_file_end": getattr(item, "source_file_end", None),
        }

    def _select(
        self,
        distances,
        k,
        *,
        tool,
        query,
        parsed_query,
        score_name,
        score_unit=None,
        lower_is_better=True,
    ):
        is_text = tool == "retrieve_from_text"
        (
            prior_indices,
            prior_order,
            effective_k,
            remaining_before,
            policy_mode,
        ) = self._selection_policy(tool, k)
        prior_evidence_sources = (
            self.get_evidence_ledger()
            if policy_mode == "unified"
            else []
        )
        evidence_state_version_before = getattr(self, "_evidence_state_version", 0)
        if not self.items:
            selected = []
            indices = np.asarray([], dtype=int)
            distance_array = np.asarray([], dtype=float)
        else:
            distance_array = np.asarray(distances, dtype=float)
            if lower_is_better:
                full_ranking = np.argsort(distance_array, kind="stable")
            else:
                full_ranking = np.argsort(-distance_array, kind="stable")
            eligible_ranking = np.asarray(
                [
                    int(index)
                    for index in full_ranking
                    if int(index) not in prior_indices
                ],
                dtype=int,
            )
            count = min(effective_k, len(eligible_ranking))
            indices = eligible_ranking[:count]
            selected = [self.items[int(index)] for index in indices]
            self.working_memory.extend(selected)

        if not self.items:
            full_ranking = np.asarray([], dtype=int)
        selected_records = []
        for rank, index in enumerate(indices, start=1):
            record = self._item_record(self.items[int(index)], int(index))
            record.update({"rank": rank, "score": float(distance_array[int(index)])})
            selected_records.append(record)

        self._commit_evidence_selection(indices, tool=tool, query=query)
        question_mode = (
            is_text
            and getattr(self, "text_episode_mode", "per_call") == "question"
            and policy_mode != "unified"
        )
        selected_after = (
            list(self._text_episode_selected_indices) if question_mode else []
        )
        question_budget = (
            int(getattr(self, "question_text_evidence_budget", self.text_k))
            if question_mode
            else None
        )
        remaining_after = (
            max(question_budget - len(selected_after), 0)
            if question_budget is not None
            else None
        )
        if policy_mode == "unified":
            selected_after = list(self._evidence_selected_indices)
            question_budget = int(self.question_evidence_budget)
            remaining_after = max(question_budget - len(selected_after), 0)

        trace_record = {
            "call_index": len(self.retrieval_trace) + 1,
            "tool": tool,
            "query": query,
            "parsed_query": parsed_query,
            "score_name": score_name,
            "score_unit": score_unit,
            "lower_is_better": bool(lower_is_better),
            "candidate_count": len(self.items),
            "requested_k": int(k),
            "effective_requested_k": int(effective_k),
            "returned_count": len(selected),
            "selected": selected_records,
            # The complete lightweight ranking lets an audit determine the exact
            # rank of a ground-truth caption without duplicating every caption.
            "ranking": [
                {
                    "rank": rank,
                    "entry_id": getattr(self.items[int(index)], "entry_id", None),
                    "local_index": int(index),
                    "score": float(distance_array[int(index)]),
                }
                for rank, index in enumerate(full_ranking, start=1)
            ],
        }
        if is_text:
            trace_record.update(
                {
                    "text_episode_mode": getattr(
                        self,
                        "text_episode_mode",
                        "per_call",
                    ),
                    "text_episode_id": getattr(self, "_text_episode_id", 0),
                    "question_evidence_budget": question_budget,
                    "question_budget_remaining_before": remaining_before,
                    "question_budget_remaining_after": remaining_after,
                    "episode_selected_entry_ids_before": [
                        getattr(self.items[index], "entry_id", None)
                        for index in prior_order
                    ],
                    "episode_selected_entry_ids_after": [
                        getattr(self.items[index], "entry_id", None)
                        for index in selected_after
                    ],
                    "budget_exhausted": (
                        remaining_after == 0
                        if remaining_after is not None
                        else False
                    ),
                }
            )
        if policy_mode == "unified":
            trace_record.update(
                {
                    "evidence_episode_mode": "unified_top1",
                    "evidence_episode_id": self._evidence_episode_id,
                    "question_evidence_budget": question_budget,
                    "question_budget_remaining_before": remaining_before,
                    "question_budget_remaining_after": remaining_after,
                    "global_selected_entry_ids_before": [
                        getattr(self.items[index], "entry_id", None)
                        for index in prior_order
                    ],
                    "global_selected_entry_ids_after": [
                        getattr(self.items[index], "entry_id", None)
                        for index in selected_after
                    ],
                    "prior_evidence_sources": prior_evidence_sources,
                    "evidence_state_version_before": evidence_state_version_before,
                    "evidence_state_version": self._evidence_state_version,
                    "budget_exhausted": remaining_after == 0,
                }
            )
        self.retrieval_trace.append(trace_record)
        return selected, trace_record

    def search_by_text(self, query: str) -> str:
        query_embedding = np.asarray(
            self.embedder.embed_query(query), dtype=np.float32
        )
        embeddings = np.asarray(self.text_embeddings)
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        selected, trace = self._select(
            distances,
            self.text_k,
            tool="retrieve_from_text",
            query=query,
            parsed_query=query,
            score_name="embedding_l2_distance",
        )
        output = self.memory_to_string(selected)
        if not selected:
            output = self.empty_text_result_context(
                bool(trace.get("budget_exhausted", False))
            )
        trace["returned_context"] = output
        return output

    def search_by_position(self, query: tuple) -> str:
        query_position = np.asarray(query, dtype=np.float32)
        positions = np.asarray([item.position for item in self.items], dtype=np.float32)
        distances = np.linalg.norm(positions - query_position, axis=1)
        selected, trace = self._select(
            distances,
            self.numeric_k,
            tool="retrieve_from_position",
            query=list(query),
            parsed_query=query_position.astype(float).tolist(),
            score_name="position_l2_distance",
            score_unit="dataset_position_unit",
        )
        output = self.memory_to_string(selected)
        if not selected and trace.get("budget_exhausted"):
            output = self.empty_text_result_context(True)
        trace["returned_context"] = output
        return output

    def search_by_time(self, hms_time: str) -> str:
        hms_time = hms_time.strip()
        template = "%m/%d/%Y %H:%M:%S"
        try:
            query_time = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple())
        except ValueError:
            date = strftime("%m/%d/%Y", localtime(self.time_offset))
            query_time = time.mktime(
                datetime.datetime.strptime(f"{date} {hms_time}", template).timetuple()
            )
        distances = [abs(item.time - query_time) for item in self.items]
        selected, trace = self._select(
            distances,
            self.numeric_k,
            tool="retrieve_from_time",
            query=hms_time,
            parsed_query={
                "unix_timestamp": float(query_time),
                "local_datetime": strftime(
                    "%Y-%m-%d %H:%M:%S", localtime(query_time)
                ),
            },
            score_name="absolute_time_delta",
            score_unit="seconds",
        )
        output = self.memory_to_string(selected)
        if not selected and trace.get("budget_exhausted"):
            output = self.empty_text_result_context(True)
        trace["returned_context"] = output
        return output

    def memory_to_string(self, memory_list):
        output = ""
        for item in memory_list:
            timestamp = strftime("%Y-%m-%d %H:%M:%S", localtime(item.time))
            output += (
                f"At time={timestamp}, the robot was at an average position of "
                f"{np.asarray(item.position).round(3).tolist()}."
                f"The robot saw the following: {item.caption}\n\n"
            )
        return output
