"""Minimal TensorRT engine wrapper for Jetson Python runtime.

Usage:
    runner = TRTRunner('/path/to/engine.engine')
    outputs = runner.infer({'image': np_array_NCHW_float16})
    # outputs is dict keyed by output binding names.

The wrapper does host<->device copies; it doesn't pin memory or use
streams. For ≥100 FPS you may want to optimise that — see notes in
jetson_deployment_workplan.md §11.

Targets the TensorRT 10 tensor-I/O API (num_io_tensors / get_tensor_name /
execute_async_v3). The previous binding-index API was removed in TRT 10.
"""
from __future__ import annotations

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTRunner:
    """Thin wrapper around a serialized TRT 10 engine."""

    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(_TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_names, self.output_names = [], []
        self.host_buffers = {}
        self.device_buffers = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            is_input = (mode == trt.TensorIOMode.INPUT)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(1 if d < 0 else d for d in shape)

            if is_input:
                self.context.set_input_shape(name, shape)
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            host = np.empty(shape, dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.host_buffers[name] = host
            self.device_buffers[name] = device
            self.context.set_tensor_address(name, int(device))

    def infer(self, inputs: dict) -> dict:
        for name, arr in inputs.items():
            if name not in self.host_buffers:
                raise KeyError(f"unknown input binding {name!r}; engine has {self.input_names}")
            target = self.host_buffers[name]
            arr = np.ascontiguousarray(arr).astype(target.dtype, copy=False)
            if arr.shape != target.shape:
                raise ValueError(f"{name}: shape {arr.shape} != expected {target.shape}")
            np.copyto(target, arr)
            cuda.memcpy_htod_async(self.device_buffers[name], target, self.stream)

        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")

        out = {}
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], self.stream)
        self.stream.synchronize()
        for name in self.output_names:
            out[name] = self.host_buffers[name].copy()
        return out

    def __repr__(self):
        return (f"TRTRunner(inputs={self.input_names}, "
                f"outputs={self.output_names})")
