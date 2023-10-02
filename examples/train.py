import os
from absl import app, flags
import datasets
import tree
import tensorflow as tf

from keras_rwkv.models.v4 import RwkvBackbone, RwkvCausalLMPreprocessor, RwkvCausalLM
from keras_rwkv.backend import keras

flags.DEFINE_integer("batch_size", 2, "Minibatch size.")
flags.DEFINE_integer("epochs", 10, "Number of epochs to train for.")
flags.DEFINE_bool("original_cuda", False, "Use original cuda implementation.")
flags.DEFINE_bool(
    "parallel_wkv", False, "Use parallel work-efficient wkv implementation"
)
flags.DEFINE_bool(
    "smoke", False, "Run a smoke test (quick run to make sure things work)"
)


def main(_):
    FLAGS = flags.FLAGS
    if (
        FLAGS.original_cuda
        and os.environ.get("USE_RANDOM_WKV_CUDA_GRADIENTS", None) is None
    ):
        os.environ["USE_RANDOM_WKV_CUDA_GRADIENTS"] = "1"
    # Load data.
    data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    data = data.filter(lambda x: len(x["text"]) > 0 and not x["text"].isspace())
    data_kwargs = {
        "batch_size": FLAGS.batch_size,
        "drop_remainder": True,
        "columns": "text",
        "label_cols": None,
        "prefetch": False,
    }
    train_data = data["train"].to_tf_dataset(shuffle=True, **data_kwargs)
    val_data, test_data = (
        data[key].to_tf_dataset(**data_kwargs) for key in ("validation", "test")
    )

    preset = "rwkv-4-pile-169m"
    backbone = RwkvBackbone.from_preset(
        preset,
        load_weights=False,
        use_original_cuda_wkv=FLAGS.original_cuda,
        parallel_wkv=FLAGS.parallel_wkv,
    )
    preprocessor = RwkvCausalLMPreprocessor.from_preset(preset)
    lm = RwkvCausalLM(backbone, preprocessor=None)

    def map_func(text):
        output = preprocessor(text)
        # ensure static shape info is there - required for jax2tf
        for el in tree.flatten(output):
            el.set_shape((FLAGS.batch_size, *el.shape[1:]))
        return output

    train_data, val_data, test_data = (
        ds.map(map_func).prefetch(tf.data.AUTOTUNE)
        for ds in (train_data, val_data, test_data)
    )

    lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        # steps_per_execution="auto",
    )
    # lm.summary()

    epochs = FLAGS.epochs
    if FLAGS.smoke:
        train_data = train_data.take(50)
        val_data = val_data.take(50)
        epochs = 2
    # lm.fit(train_data, validation_data=val_data, epochs=epochs)
    lm.fit(train_data, epochs=epochs)


if __name__ == "__main__":
    app.run(main)
