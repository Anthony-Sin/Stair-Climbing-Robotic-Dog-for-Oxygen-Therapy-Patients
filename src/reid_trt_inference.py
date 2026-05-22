from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


class ReIDTRTInference:
    def __init__(self, engine_path: str):
        engine_file = Path(engine_path)
        if not engine_file.is_file():
            raise FileNotFoundError(f"ReID TensorRT engine not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        # Required for engines that rely on TensorRT plugin layers.
        trt.init_libnvinfer_plugins(logger, "")
        with engine_file.open("rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.input_binding_idx = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        self.output_binding_idx = next(i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_binding_idx))

        self.stream = cuda.Stream()
        self.bindings = [0] * self.engine.num_bindings
        self.d_input = None
        self.d_output = None
        self.input_size_bytes = 0
        self.output_size_bytes = 0
        self.embedding_dim: Optional[int] = None

    def preprocess_bgr_crops(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        processed: List[np.ndarray] = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        target_w, target_h = 128, 256
        pad_value = tuple(int(round(v * 255.0)) for v in mean)

        for crop in crops:
            if crop is None or crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(target_w / max(1, w), target_h / max(1, h))
            resized_w = max(1, min(target_w, int(round(w * scale))))
            resized_h = max(1, min(target_h, int(round(h * scale))))
            rgb = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

            # Pad with the normalization mean so the border becomes approximately zero after normalization.
            pad_left = (target_w - resized_w) // 2
            pad_right = target_w - resized_w - pad_left
            pad_top = (target_h - resized_h) // 2
            pad_bottom = target_h - resized_h - pad_top
            rgb = cv2.copyMakeBorder(
                rgb,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=pad_value,
            )
            x = rgb.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            processed.append(x)

        if not processed:
            return np.empty((0, 3, 256, 128), dtype=np.float32)

        return np.ascontiguousarray(np.stack(processed, axis=0).astype(np.float32))

    def _ensure_buffers(self, input_shape: tuple) -> tuple:
        self.context.set_binding_shape(self.input_binding_idx, input_shape)
        output_shape = tuple(self.context.get_binding_shape(self.output_binding_idx))

        input_bytes = int(np.prod(input_shape) * np.dtype(self.input_dtype).itemsize)
        output_bytes = int(np.prod(output_shape) * np.dtype(self.output_dtype).itemsize)

        if self.d_input is None or input_bytes != self.input_size_bytes:
            self.d_input = cuda.mem_alloc(input_bytes)
            self.input_size_bytes = input_bytes

        if self.d_output is None or output_bytes != self.output_size_bytes:
            self.d_output = cuda.mem_alloc(output_bytes)
            self.output_size_bytes = output_bytes

        self.bindings[self.input_binding_idx] = int(self.d_input)
        self.bindings[self.output_binding_idx] = int(self.d_output)

        if len(output_shape) >= 2 and output_shape[-1] > 0:
            self.embedding_dim = int(output_shape[-1])

        return output_shape

    def infer_embeddings(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape={batch.shape}")

        if batch.shape[0] == 0:
            dim = self.embedding_dim or 0
            return np.empty((0, dim), dtype=np.float32)

        input_tensor = np.ascontiguousarray(batch.astype(self.input_dtype, copy=False))
        output_shape = self._ensure_buffers(tuple(input_tensor.shape))
        host_output = np.empty(output_shape, dtype=self.output_dtype)

        cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(host_output, self.d_output, self.stream)
        self.stream.synchronize()

        return host_output.astype(np.float32, copy=False)

    def infer_crops(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        batch = self.preprocess_bgr_crops(crops)
        return self.infer_embeddings(batch)
