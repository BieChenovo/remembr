"""Small in-process vector memory for NaVQA evaluation.

The evaluation sequences contain at most hundreds of caption entries, so a
NumPy scan is simpler and more reproducible than requiring a Milvus service.
Caption embeddings generated offline are reused; only text queries are encoded.
"""

import datetime
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

    def insert(self, item: MemoryItem, text_embedding=None):
        self.items.append(item)
        if text_embedding is None:
            text_embedding = self.embedder.embed_query(item.caption)
        self.text_embeddings.append(np.asarray(text_embedding, dtype=np.float32))

    def get_working_memory(self):
        return self.working_memory

    def _select(self, distances, k):
        if not self.items:
            return []
        count = min(k, len(self.items))
        indices = np.argsort(np.asarray(distances))[:count]
        selected = [self.items[int(index)] for index in indices]
        self.working_memory.extend(selected)
        return selected

    def search_by_text(self, query: str) -> str:
        query_embedding = np.asarray(
            self.embedder.embed_query(query), dtype=np.float32
        )
        embeddings = np.asarray(self.text_embeddings)
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        return self.memory_to_string(self._select(distances, self.text_k))

    def search_by_position(self, query: tuple) -> str:
        query_position = np.asarray(query, dtype=np.float32)
        positions = np.asarray([item.position for item in self.items], dtype=np.float32)
        distances = np.linalg.norm(positions - query_position, axis=1)
        return self.memory_to_string(self._select(distances, self.numeric_k))

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
        return self.memory_to_string(self._select(distances, self.numeric_k))

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
