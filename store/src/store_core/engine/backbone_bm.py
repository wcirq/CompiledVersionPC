from typing import Optional, Tuple

import numpy as np
import torch


class BMFeatureBackbone:
    def __init__(
        self,
        bmodel_path: str,
        device_id: int = 0,
        graph_name: Optional[str] = None,
        input_name: Optional[str] = None,
        feat2_output_name: Optional[str] = None,
        feat3_output_name: Optional[str] = None,
    ):
        if not bmodel_path:
            raise ValueError("bm backbone requires a valid --backbone_bmodel_path.")

        self.backend = "bm"
        self.bmodel_path = str(bmodel_path)
        self.device_id = int(device_id)
        self.graph_name = graph_name
        self.input_name = input_name
        self.feat2_output_name = feat2_output_name
        self.feat3_output_name = feat3_output_name

        self._sail = None
        self.engine = None
        self.max_batch: Optional[int] = None
        self.input_hw: Optional[Tuple[int, int]] = None

        self._init_runtime()

    def _init_runtime(self):
        try:
            import sophon.sail as sail
        except ImportError as exc:
            raise ImportError(
                "bm backbone requires sophon.sail. Please install the Sophon runtime on the BM1684X host."
            ) from exc

        self._sail = sail
        self.engine = sail.Engine(self.bmodel_path, self.device_id, sail.IOMode.SYSIO)

        graph_names = self.engine.get_graph_names()
        if not graph_names:
            raise RuntimeError(f"No graph found in bmodel: {self.bmodel_path}")
        if self.graph_name is None:
            self.graph_name = graph_names[0]

        input_names = list(self.engine.get_input_names(self.graph_name))
        if not input_names:
            raise RuntimeError("BM backbone graph has no inputs.")
        if self.input_name is None:
            self.input_name = input_names[0]
        if self.input_name not in input_names:
            raise RuntimeError(
                f"BM backbone input {self.input_name!r} missing. Available inputs: {input_names}"
            )

        output_names = list(self.engine.get_output_names(self.graph_name))
        if len(output_names) < 2:
            raise RuntimeError(
                "BM backbone expects at least two outputs for feat2 and feat3."
            )

        if self.feat2_output_name is None:
            self.feat2_output_name = output_names[0]
        if self.feat3_output_name is None:
            remaining = [name for name in output_names if name != self.feat2_output_name]
            self.feat3_output_name = remaining[0] if remaining else output_names[-1]

        missing_outputs = [
            name for name in (self.feat2_output_name, self.feat3_output_name) if name not in output_names
        ]
        if missing_outputs:
            raise RuntimeError(
                f"BM backbone outputs missing: {missing_outputs}. Available outputs: {output_names}"
            )

        input_shape = tuple(self.engine.get_input_shape(self.graph_name, self.input_name))
        if len(input_shape) != 4:
            raise RuntimeError(
                f"BM backbone expects a 4D input tensor, got shape {input_shape}."
            )

        batch_size = int(input_shape[0])
        self.max_batch = batch_size if batch_size > 0 else None
        channels = int(input_shape[1])
        if channels != 3:
            raise RuntimeError(
                f"BM backbone expects 3-channel input, got shape {input_shape}."
            )
        self.input_hw = (int(input_shape[2]), int(input_shape[3]))

    def eval(self):
        return self

    def __call__(self, images: torch.Tensor):
        if self.engine is None:
            raise ValueError("BM backbone engine not initialized.")

        arr = images.detach().cpu().float().numpy().astype(np.float32, copy=False)
        if arr.ndim != 4:
            raise ValueError(f"BM backbone expects NCHW input, got shape {arr.shape}.")
        if self.input_hw is not None and tuple(arr.shape[2:]) != tuple(self.input_hw):
            raise ValueError(
                f"BM backbone expects input HW {self.input_hw}, got {tuple(arr.shape[2:])}."
            )

        feat2_chunks = []
        feat3_chunks = []
        chunk_size = self.max_batch or int(arr.shape[0])

        for st in range(0, arr.shape[0], chunk_size):
            ed = min(st + chunk_size, arr.shape[0])
            chunk = np.ascontiguousarray(arr[st:ed])
            input_arr = self._prepare_input(chunk)
            outputs = self.engine.process(self.graph_name, {self.input_name: input_arr})
            feat2_chunks.append(self._slice_output(outputs, self.feat2_output_name, ed - st))
            feat3_chunks.append(self._slice_output(outputs, self.feat3_output_name, ed - st))

        feat2 = torch.from_numpy(np.concatenate(feat2_chunks, axis=0)).float()
        feat3 = torch.from_numpy(np.concatenate(feat3_chunks, axis=0)).float()
        return feat2, feat3

    def _prepare_input(self, arr: np.ndarray) -> np.ndarray:
        if self.max_batch is None or arr.shape[0] == self.max_batch:
            return np.ascontiguousarray(arr, dtype=np.float32)
        if arr.shape[0] > self.max_batch:
            raise ValueError(
                f"BM backbone chunk size {arr.shape[0]} exceeds model batch size {self.max_batch}."
            )
        padded = np.zeros((self.max_batch, *arr.shape[1:]), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded

    @staticmethod
    def _slice_output(outputs, output_name: str, rows: int) -> np.ndarray:
        if output_name not in outputs:
            raise RuntimeError(
                f"BM backbone output {output_name!r} missing. Available outputs: {list(outputs.keys())}"
            )
        arr = np.asarray(outputs[output_name], dtype=np.float32)
        if arr.ndim >= 1 and arr.shape[0] >= rows:
            return np.ascontiguousarray(arr[:rows], dtype=np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)
