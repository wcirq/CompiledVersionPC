from typing import Optional, Tuple

import numpy as np
import torch


class BMVectorIndex:
    def __init__(
        self,
        bmodel_path: str,
        n_neighbors: int = 1,
        device_id: int = 0,
        query_chunk_size: int = 256,
        db_chunk_size: int = 4096,
        graph_name: Optional[str] = None,
        query_input_name: Optional[str] = None,
        database_input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        if not bmodel_path:
            raise ValueError("bm backend requires a valid --bm_bmodel_path.")

        self.bmodel_path = str(bmodel_path)
        self.n_neighbors = int(n_neighbors)
        self.device_id = int(device_id)
        self.query_chunk_size = int(query_chunk_size)
        self.db_chunk_size = int(db_chunk_size)
        self.graph_name = graph_name
        self.query_input_name = query_input_name
        self.database_input_name = database_input_name
        self.output_name = output_name

        self._sail = None
        self.engine = None
        self.memory_bank: Optional[np.ndarray] = None
        self.memory_bank_norms: Optional[np.ndarray] = None
        self.embed_dim: Optional[int] = None
        self.max_query_batch: Optional[int] = None
        self.max_db_batch: Optional[int] = None

        self._init_runtime()

    def _init_runtime(self):
        try:
            import sophon.sail as sail
        except ImportError as exc:
            raise ImportError(
                "bm backend requires sophon.sail. Please install the Sophon runtime on the BM1684 host."
            ) from exc

        self._sail = sail
        self.engine = sail.Engine(self.bmodel_path, self.device_id, sail.IOMode.SYSIO)

        graph_names = self.engine.get_graph_names()
        if not graph_names:
            raise RuntimeError(f"No graph found in bmodel: {self.bmodel_path}")
        if self.graph_name is None:
            self.graph_name = graph_names[0]

        input_names = list(self.engine.get_input_names(self.graph_name))
        if len(input_names) < 2:
            raise RuntimeError(
                "BM vector backend expects a bmodel graph with at least two inputs: queries and database."
            )

        if self.query_input_name is None:
            self.query_input_name = input_names[0]
        if self.database_input_name is None:
            for name in input_names:
                if name != self.query_input_name:
                    self.database_input_name = name
                    break
        if self.database_input_name is None:
            raise RuntimeError("Unable to resolve BM database input name.")

        output_names = list(self.engine.get_output_names(self.graph_name))
        if not output_names:
            raise RuntimeError("BM vector backend graph has no outputs.")
        if self.output_name is None:
            self.output_name = output_names[0]

        query_shape = tuple(self.engine.get_input_shape(self.graph_name, self.query_input_name))
        db_shape = tuple(self.engine.get_input_shape(self.graph_name, self.database_input_name))
        self.max_query_batch, query_dim = self._parse_2d_shape(query_shape, "query")
        self.max_db_batch, db_dim = self._parse_2d_shape(db_shape, "database")
        if query_dim != db_dim:
            raise RuntimeError(
                f"BM vector backend expects matching embedding dims, got query dim {query_dim} and database dim {db_dim}."
            )
        self.embed_dim = query_dim

    @staticmethod
    def _parse_2d_shape(shape: Tuple[int, ...], label: str) -> Tuple[Optional[int], int]:
        if len(shape) != 2:
            raise RuntimeError(f"BM vector backend expects {label} input shape to be 2D, got {shape}.")
        batch_size = shape[0] if int(shape[0]) > 0 else None
        embed_dim = int(shape[1])
        if embed_dim <= 0:
            raise RuntimeError(f"BM vector backend requires static embedding dimension for {label} input, got {shape}.")
        return batch_size, embed_dim

    def fit(self, memory_bank: torch.Tensor):
        if memory_bank is None or memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
            raise ValueError("Invalid memory bank.")

        mb = memory_bank.detach().cpu().float().numpy().astype(np.float32, copy=False)
        if self.embed_dim is not None and int(mb.shape[1]) != int(self.embed_dim):
            raise ValueError(
                f"BM vector backend expects embedding dim {self.embed_dim}, got {int(mb.shape[1])}."
            )

        self.memory_bank = np.ascontiguousarray(mb)
        self.memory_bank_norms = np.sum(self.memory_bank * self.memory_bank, axis=1, dtype=np.float32)

    def kneighbors(self, queries: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.memory_bank is None or self.memory_bank_norms is None:
            raise ValueError("BM vector backend not initialized.")

        q = queries.detach().cpu().float().numpy().astype(np.float32, copy=False)
        if q.ndim != 2:
            raise ValueError("queries must be a 2D tensor.")
        if self.embed_dim is not None and int(q.shape[1]) != int(self.embed_dim):
            raise ValueError(f"BM vector backend expects query dim {self.embed_dim}, got {int(q.shape[1])}.")

        n_queries = int(q.shape[0])
        k = min(max(1, self.n_neighbors), int(self.memory_bank.shape[0]))
        best_dist = np.full((n_queries, k), np.inf, dtype=np.float32)
        best_idx = np.full((n_queries, k), -1, dtype=np.int64)

        query_chunk = max(1, self.query_chunk_size)
        if self.max_query_batch is not None:
            query_chunk = min(query_chunk, self.max_query_batch)
        db_chunk = max(1, self.db_chunk_size)
        if self.max_db_batch is not None:
            db_chunk = min(db_chunk, self.max_db_batch)

        query_norms = np.sum(q * q, axis=1, dtype=np.float32)

        for q_st in range(0, n_queries, query_chunk):
            q_ed = min(q_st + query_chunk, n_queries)
            q_chunk = np.ascontiguousarray(q[q_st:q_ed])
            q_norm_chunk = query_norms[q_st:q_ed]
            local_best_dist = np.full((q_ed - q_st, k), np.inf, dtype=np.float32)
            local_best_idx = np.full((q_ed - q_st, k), -1, dtype=np.int64)

            for db_st in range(0, self.memory_bank.shape[0], db_chunk):
                db_ed = min(db_st + db_chunk, self.memory_bank.shape[0])
                db_chunk_arr = np.ascontiguousarray(self.memory_bank[db_st:db_ed])
                dot = self._run_similarity(q_chunk, db_chunk_arr)
                dist2 = q_norm_chunk[:, None] + self.memory_bank_norms[db_st:db_ed][None, :] - 2.0 * dot
                dist2 = np.maximum(dist2, 0.0, out=dist2)

                candidate_dist = np.sqrt(dist2, dtype=np.float32)
                candidate_idx = np.broadcast_to(
                    np.arange(db_st, db_ed, dtype=np.int64)[None, :],
                    candidate_dist.shape,
                )
                local_best_dist, local_best_idx = self._merge_topk(
                    local_best_dist,
                    local_best_idx,
                    candidate_dist,
                    candidate_idx,
                    k,
                )

            best_dist[q_st:q_ed] = local_best_dist
            best_idx[q_st:q_ed] = local_best_idx

        return best_dist.astype(np.float32), best_idx.astype(np.int64)

    def _run_similarity(self, queries: np.ndarray, database: np.ndarray) -> np.ndarray:
        if self.engine is None:
            raise ValueError("BM engine not initialized.")

        query_input = self._prepare_input(queries, self.max_query_batch, self.embed_dim)
        db_input = self._prepare_input(database, self.max_db_batch, self.embed_dim)
        outputs = self.engine.process(
            self.graph_name,
            {
                self.query_input_name: query_input,
                self.database_input_name: db_input,
            },
        )
        if self.output_name not in outputs:
            raise RuntimeError(
                f"BM vector backend output {self.output_name!r} missing. Available outputs: {list(outputs.keys())}"
            )
        output = np.asarray(outputs[self.output_name], dtype=np.float32)
        return self._reshape_similarity_output(output, queries.shape[0], database.shape[0])

    @staticmethod
    def _prepare_input(arr: np.ndarray, max_batch: Optional[int], embed_dim: Optional[int]) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"BM vector backend expects 2D input arrays, got shape {arr.shape}.")
        if embed_dim is not None and int(arr.shape[1]) != int(embed_dim):
            raise ValueError(f"BM vector backend expects embedding dim {embed_dim}, got {int(arr.shape[1])}.")
        if max_batch is None or arr.shape[0] == max_batch:
            return np.ascontiguousarray(arr)
        if arr.shape[0] > max_batch:
            raise ValueError(f"BM vector backend chunk size {arr.shape[0]} exceeds model batch size {max_batch}.")
        padded = np.zeros((max_batch, arr.shape[1]), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded

    @staticmethod
    def _reshape_similarity_output(output: np.ndarray, q_rows: int, db_rows: int) -> np.ndarray:
        if output.ndim == 2:
            sim = output
        else:
            squeezed = np.squeeze(output)
            if squeezed.ndim != 2:
                raise RuntimeError(
                    f"BM vector backend expects a matrix-like output, got raw output shape {output.shape}."
                )
            sim = squeezed
        if sim.shape[0] < q_rows or sim.shape[1] < db_rows:
            raise RuntimeError(
                f"BM vector backend output shape {sim.shape} is smaller than required ({q_rows}, {db_rows})."
            )
        return np.ascontiguousarray(sim[:q_rows, :db_rows], dtype=np.float32)

    @staticmethod
    def _merge_topk(
        current_dist: np.ndarray,
        current_idx: np.ndarray,
        candidate_dist: np.ndarray,
        candidate_idx: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        merged_dist = np.concatenate([current_dist, candidate_dist], axis=1)
        merged_idx = np.concatenate([current_idx, candidate_idx], axis=1)
        order = np.argpartition(merged_dist, kth=k - 1, axis=1)[:, :k]
        top_dist = np.take_along_axis(merged_dist, order, axis=1)
        top_idx = np.take_along_axis(merged_idx, order, axis=1)
        sort_order = np.argsort(top_dist, axis=1)
        top_dist = np.take_along_axis(top_dist, sort_order, axis=1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=1)
        return top_dist.astype(np.float32), top_idx.astype(np.int64)
