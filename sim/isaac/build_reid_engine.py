import tensorrt as trt
import onnx

# Validate the ONNX file first
model = onnx.load('/models/reid/osnet_ain_x1_0.onnx')
onnx.checker.check_model(model)
print(f'ONNX model opset: {model.opset_import[0].version}')

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open('/models/reid/osnet_ain_x1_0.onnx', 'rb') as f:
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

input_tensor = network.get_input(0)
input_shape = tuple(input_tensor.shape)
if any(dim < 0 for dim in input_shape):
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 256, 128),
        opt=(8, 3, 256, 128),
        max=(16, 3, 256, 128),
    )
    config.add_optimization_profile(profile)
    print(f'Added dynamic batch profile for {input_tensor.name}: min=1 opt=8 max=16')
else:
    print(f'Using static ReID input shape from ONNX: {input_shape}')

print('Building ReID engine...')
engine_bytes = builder.build_serialized_network(network, config)

if engine_bytes is None:
    raise RuntimeError('build_serialized_network returned None — check TRT logs above')

with open('/models/reid/osnet_ain_x1_0.trt', 'wb') as f:
    f.write(engine_bytes)
print('Done: /models/reid/osnet_ain_x1_0.trt')
