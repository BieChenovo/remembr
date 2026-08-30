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

    def __init__(self, embedding_model, time_offset, text_k=5, numeric_k=4):
        if embedding_model not in self._embedder_cache:
            self._embedder_cache[embedding_model] = HuggingFaceEmbeddings(
                model_name=embedding_model
            )
        self.embedder = self._embedder_cache[embedding_model]
        self.time_offset = time_offset
        self.text_k = text_k
        self.numeric_k = numeric_k
        self.reset()

    def reset(self):
        self.items = []
        self.text_embeddings = []
        self.working_memory = []
        self.retrieval_trace = []
        self.candidate_pool_metadata = {}

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
        if not self.items:
            selected = []
            indices = np.asarray([], dtype=int)
            distance_array = np.asarray([], dtype=float)
        else:
            count = min(k, len(self.items))
            distance_array = np.asarray(distances, dtype=float)
            if lower_is_better:
                full_ranking = np.argsort(distance_array, kind="stable")
            else:
                full_ranking = np.argsort(-distance_array, kind="stable")
            indices = full_ranking[:count]
            selected = [self.items[int(index)] for index in indices]
            self.working_memory.extend(selected)

        if not self.items:
            full_ranking = np.asarray([], dtype=int)
        selected_records = []
        for rank, index in enumerate(indices, start=1):
            record = self._item_record(self.items[int(index)], int(index))
            record.update({"rank": rank, "score": float(distance_array[int(index)])})
            selected_records.append(record)

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
