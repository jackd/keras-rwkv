from keras_nlp.samplers import RandomSampler
from keras_rwkv.models.v4 import RwkvCausalLM, RwkvCausalLMPreprocessor, RwkvBackbone
from keras_rwkv.backend import ops

preset = "rwkv-4-pile-169m"
# preset = "rwkv-4-pile-430m"

backbone = RwkvBackbone.from_preset(
    preset, use_original_cuda_wkv=False, parallel_wkv=True
)
preprocessor = RwkvCausalLMPreprocessor.from_preset(preset)

lm = RwkvCausalLM(backbone, preprocessor)
lm.backbone.summary()
tokenizer = lm.preprocessor.tokenizer
sampler = RandomSampler()
# sampler = "greedy"
lm.compile(sampler=sampler)

ctx = """\nIn a shocking finding, scientist discovered a herd of dragons living in a \
remote, previously unexplored valley, in Tibet. Even more surprising to the \
researchers was the fact that the dragons spoke perfect Chinese."""
output = lm.generate([ctx])
print(ops.convert_to_numpy(output[0]).item())
