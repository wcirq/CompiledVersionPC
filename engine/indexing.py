from typing import Optional, Tuple

try:
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors
except Exception as exc:
    from .debug_utils import print_exception_details

    print_exception_details(exc, context="engine.indexing import failed")
    raise

from .debug_utils import guarded, print_exception_details


class MemoryIndex:
    @guarded("MemoryIndex.__init__ failed")
    def __init__(
        self,
        backend: str = "auto",
        n_neighbors: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        query_chunk_size: int = 8192,
    ):
        self.backend = backend
        self.n_neighbors = int(n_neighbors)
        self.device = device
        self.query_chunk_size = int(query_chunk_size)
        self.memory_bank_torch: Optional[torch.Tensor] = None
        self.sklearn_index = None
        self.faiss_index = None

    @guarded("MemoryIndex._resolve_backend failed")
    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if torch.cuda.is_available() and "cuda" in self.device:
            return "torch"
        return "sklearn"

    @guarded("MemoryIndex.fit failed")
    def fit(self, memory_bank: torch.Tensor):
        if memory_bank is None or memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
            raise ValueError("Invalid memory bank.")

        backend = self._resolve_backend()
        mb_cpu = memory_bank.detach().cpu().float().numpy()

        if backend == "torch":
            self.memory_bank_torch = memory_bank.detach().to(self.device, dtype=torch.float32)
            self.sklearn_index = None
            self.faiss_index = None
            return

        if backend == "faiss":
            try:
                import faiss

                index = faiss.IndexFlatL2(int(mb_cpu.shape[1]))
                index.add(mb_cpu.astype(np.float32))
                self.faiss_index = index
                self.memory_bank_torch = None
                self.sklearn_index = None
                return
            except Exception as exc:
                print_exception_details(exc, context="MemoryIndex.fit faiss backend unavailable")
                print(f"[WARN] faiss backend unavailable, fallback to sklearn: {exc}")

        self.sklearn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean",
        )
        self.sklearn_index.fit(mb_cpu)
        self.memory_bank_torch = None
        self.faiss_index = None

    @torch.no_grad()
    @guarded("MemoryIndex.kneighbors failed")
    def kneighbors(self, queries: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        backend = self._resolve_backend()

        if backend == "torch" and self.memory_bank_torch is not None:
            return self._kneighbors_torch(queries)

        if backend == "faiss" and self.faiss_index is not None:
            q = queries.detach().cpu().float().numpy()
            d2, inds = self.faiss_index.search(q, self.n_neighbors)
            return np.sqrt(np.maximum(d2, 0.0), dtype=np.float32), inds.astype(np.int64)

        if self.sklearn_index is None:
            raise ValueError("NN index not built.")
        q = queries.detach().cpu().float().numpy()
        distances, inds = self.sklearn_index.kneighbors(q)
        return distances.astype(np.float32), inds.astype(np.int64)

    @torch.no_grad()
    @guarded("MemoryIndex._kneighbors_torch failed")
    def _kneighbors_torch(self, queries: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.memory_bank_torch is None:
            raise ValueError("Torch backend not initialized.")

        q = queries.detach().to(self.device, dtype=torch.float32)
        db = self.memory_bank_torch
        all_d = []
        all_i = []
        chunk = max(1, self.query_chunk_size)

        for st in range(0, q.shape[0], chunk):
            ed = min(st + chunk, q.shape[0])
            dist = torch.cdist(q[st:ed], db, p=2)
            if self.n_neighbors == 1:
                d, i = torch.min(dist, dim=1, keepdim=True)
            else:
                d, i = torch.topk(dist, k=self.n_neighbors, dim=1, largest=False, sorted=True)
            all_d.append(d.detach().cpu())
            all_i.append(i.detach().cpu())

        return (
            torch.cat(all_d, dim=0).numpy().astype(np.float32),
            torch.cat(all_i, dim=0).numpy().astype(np.int64),
        )
