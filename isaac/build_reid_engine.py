import tensorrt as trt
import onnx

# Validate the ONNX file first
model = onnx.load('/models/osnet_ain_x1_0.onnx')
onnx.checker.check_model(model)
print(f'ONNX model opset: {model.opset_import[0].version}')

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open('/models/osnet_ain_x1_0.onnx', 'rb') as f:
    success = parser.parse(f.read())

if not success:
    for i in range(parser.num_errors):
        err = parser.get_error(i)
        print(f'[TRT parse error {i}] code={err.code()} node={err.node()} desc={err.desc()}')
    raise RuntimeError('ONNX parsing failed — see errors above')

print(f'Network: {network.num_layers} layers, {network.num_outputs} output(s)')

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

print('Building ReID engine...')
engine_bytes = builder.build_serialized_network(network, config)

if engine_bytes is None:
    raise RuntimeError('build_serialized_network returned None — check TRT logs above')

with open('/models/osnet_ain_x1_0.trt', 'wb') as f:
    f.write(engine_bytes)
print('Done: /models/osnet_ain_x1_0.trt')