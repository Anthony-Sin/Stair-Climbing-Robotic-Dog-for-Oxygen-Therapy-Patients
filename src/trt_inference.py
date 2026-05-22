import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, engine_path, verbose=False):
        self.verbose = bool(verbose)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_binding_idx = next(i for i in range(self.engine.num_bindings)
                                      if self.engine.binding_is_input(i))
        self.output_binding_idx = next(i for i in range(self.engine.num_bindings)
                                       if not self.engine.binding_is_input(i))

        if self.verbose:
            print("[DEBUG] Input binding name:", self.engine.get_binding_name(self.input_binding_idx))
            print("[DEBUG] Output binding name:", self.engine.get_binding_name(self.output_binding_idx))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_binding_idx))

        # Get tensor shapes
        input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        output_shape = self.engine.get_binding_shape(self.output_binding_idx)

        if self.verbose:
            print("[DEBUG] TRT input shape:", input_shape)
            print("[DEBUG] TRT output shape:", output_shape)

        self.input_size_bytes = np.dtype(self.input_dtype).itemsize * np.prod(input_shape)
        self.output_size_bytes = np.dtype(self.output_dtype).itemsize * np.prod(output_shape)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(int(self.input_size_bytes))
        self.d_output = cuda.mem_alloc(int(self.output_size_bytes))

        # Set bindings
        self.stream = cuda.Stream()
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_binding_idx] = int(self.d_input)
        self.bindings[self.output_binding_idx] = int(self.d_output)

    def infer(self, input_tensor, debug=False):
        input_tensor_np = np.ascontiguousarray(input_tensor)

        input_shape = input_tensor_np.shape
        self.context.set_binding_shape(self.input_binding_idx, input_shape)

        input_bytes = input_tensor_np.nbytes
        output_shape = tuple(self.context.get_binding_shape(self.output_binding_idx))
        output_bytes = np.prod(output_shape) * np.dtype(self.output_dtype).itemsize

        # Allocate memory if size changed
        if self.d_input is None or input_bytes != self.input_size_bytes:
            self.d_input = cuda.mem_alloc(int(input_bytes))  # FIXED
            self.input_size_bytes = input_bytes

        if self.d_output is None or output_bytes != self.output_size_bytes:
            self.d_output = cuda.mem_alloc(int(output_bytes))  # FIXED
            self.output_size_bytes = output_bytes

        self.bindings[self.input_binding_idx] = int(self.d_input)
        self.bindings[self.output_binding_idx] = int(self.d_output)

        host_output = np.empty(output_shape, dtype=self.output_dtype)

        try:
            cuda.memcpy_htod_async(self.d_input, input_tensor_np, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(host_output, self.d_output, self.stream)
            self.stream.synchronize()
            
            if debug:
                print("[DEBUG] TRT output:")
                print("  shape:", host_output.shape)
                print("  dtype:", host_output.dtype)
                print("  min/max:", np.min(host_output), np.max(host_output))
                print("  any NaN:", np.isnan(host_output).any())

        except cuda.LogicError as e:
            print(f"[YOLO TRT] CUDA error during inference: {e}", file=sys.stderr)
            return None

        return host_output
