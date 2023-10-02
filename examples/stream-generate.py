import tree
from keras_nlp.samplers import RandomSampler
from keras_rwkv.models.v4 import RwkvCausalLM, RwkvCausalLMPreprocessor, RwkvBackbone
from keras_rwkv.backend import ops
from keras_rwkv.models.cached import call_and_create_cache, call_with_cache

preset = "rwkv-4-pile-169m"
# preset = "rwkv-4-pile-430m"

initialize_with_block_computation = True

backbone = RwkvBackbone.from_preset(
    preset, use_original_cuda_wkv=False, parallel_wkv=True
)
preprocessor = RwkvCausalLMPreprocessor.from_preset(preset)
lm = RwkvCausalLM(backbone, preprocessor)
tokenizer = preprocessor.tokenizer
sampler = RandomSampler()
# sampler = "greedy"

ctx = """\nIn a shocking finding, scientist discovered a herd of dragons living in a \
remote, previously unexplored valley, in Tibet. Even more surprising to the \
researchers was the fact that the dragons spoke perfect Chinese."""

inp = preprocessor([ctx])[0]  # pylint:disable=not-callable
token_ids = inp["token_ids"]
padding_mask = inp["padding_mask"]

current_index = ops.min(ops.count_nonzero(padding_mask, axis=1), axis=0) - 1
current_index = ops.convert_to_numpy(current_index).item()

if initialize_with_block_computation:
    logits, cache = call_and_create_cache(
        lm, inp, current_index=current_index, max_length=token_ids.shape[1]
    )
    logits = ops.expand_dims(logits[:, current_index], axis=1)
else:
    # You can also generate the cache token-by-token like below
    logits, cache = call_and_create_cache(
        lm, inp, current_index=0, max_length=token_ids.shape[1]
    )

    for i in range(1, current_index + 1):
        logits, cache = call_with_cache(
            lm,
            tree.map_structure(lambda x: ops.expand_dims(x[:, i], 1), inp),
            cache=cache,
            current_index=i,
        )


def sample(logits):
    logits = ops.squeeze(logits, axis=1)
    return sampler.get_next_token(sampler.compute_probabilities(logits))


next_token = sample(logits)
print(ctx, end="", flush=True)

while ops.convert_to_numpy(next_token).squeeze().item() != tokenizer.end_token_id:
    token_bytes = ops.convert_to_numpy(tokenizer.detokenize(next_token)).item()
    print(token_bytes.decode(), end="", flush=True)
    current_index += 1
    inp = {
        "token_ids": ops.expand_dims(next_token, 1),
        "padding_mask": ops.ones((1, 1), dtype="bool"),
    }
    logits, cache = call_with_cache(lm, inp, cache=cache, current_index=current_index)
    next_token = sample(logits)
print("\n")
