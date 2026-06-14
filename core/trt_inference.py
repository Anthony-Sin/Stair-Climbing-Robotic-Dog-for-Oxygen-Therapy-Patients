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

        # Modern TensorRT API
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)

        self.input_size_bytes = np.dtype(self.input_dtype).itemsize * np.prod(input_shape)
        self.output_size_bytes = np.dtype(self.output_dtype).itemsize * np.prod(output_shape)

        self.d_input = cuda.mem_alloc(int(self.input_size_bytes))
        self.d_output = cuda.mem_alloc(int(self.output_size_bytes))

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.stream = cuda.Stream()
    def infer(self, input_tensor, debug=False):
        input_tensor_np = np.ascontiguousarray(input_tensor)

        input_shape = input_tensor_np.shape
        self.context.set_input_shape(self.input_name, input_shape)

        input_bytes = input_tensor_np.nbytes
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_bytes = np.prod(output_shape) * np.dtype(self.output_dtype).itemsize

        # Allocate memory if size changed
        if self.d_input is None or input_bytes != self.input_size_bytes:
            self.d_input = cuda.mem_alloc(int(input_bytes))
            self.input_size_bytes = input_bytes
            self.context.set_tensor_address(self.input_name, int(self.d_input))

        if self.d_output is None or output_bytes != self.output_size_bytes:
            self.d_output = cuda.mem_alloc(int(output_bytes))
            self.output_size_bytes = output_bytes
            self.context.set_tensor_address(self.output_name, int(self.d_output))

        host_output = np.empty(output_shape, dtype=self.output_dtype)

        try:
            cuda.memcpy_htod_async(self.d_input, input_tensor_np, self.stream)
            # Use execute_async_v3 for modern TensorRT API
            self.context.execute_async_v3(stream_handle=self.stream.handle)
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