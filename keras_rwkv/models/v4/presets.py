# pylint:disable=line-too-long
backbone_presets = {
    "rwkv-4-pile-169m": {
        "metadata": {
            "description": "12-layer RWKV model trained on Pile for 322B tokens",
            "params": 169_342_464,
            "official_name": "RWKV",
            "path": "rwkv",
            "model_card": "https://huggingface.co/BlinkDL/rwkv-4-pile-169m",
        },
        "config": {
            "vocabulary_size": 50277,
            "hidden_dim": 768,
            "num_layers": 12,
            "ffn_pre": False,
        },
        "max_sequence_length": 1024,
        "weights_url": "https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth",
        "weights_hash": "7486e318c7097576ba2967ed6738ebce",
        "tokenizer_url": "https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4/20B_tokenizer.json",
        "tokenizer_hash": "56ac4821e129d2c520fdaba60abd920fa852ada51b45c0dd52bbb6bd8c985ade",
    },
    "rwkv-4-pile-430m": {
        "metadata": {
            "description": "24-layer RWKV model trained on Pile",
            "params": 430_397_440,
            "official_name": "RWKV",
            "path": "rwkv",
            "model_card": "https://huggingface.co/BlinkDL/rwkv-4-pile-430m",
        },
        "config": {
            "vocabulary_size": 50277,
            "hidden_dim": 1024,
            "num_layers": 24,
            "ffn_pre": False,
        },
        "max_sequence_length": 1024,
        "weights_url": "https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth",
        "weights_hash": "261e6b8fef1c7c9e08a4dde31bf5caf8e79c4da38126d77977a4707de82a7f64",
        "tokenizer_url": "https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4/20B_tokenizer.json",
        "tokenizer_hash": "56ac4821e129d2c520fdaba60abd920fa852ada51b45c0dd52bbb6bd8c985ade",
    },
}
