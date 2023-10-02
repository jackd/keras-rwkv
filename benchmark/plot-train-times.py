"""
This script generates the training time plots used in `performance.md`

Results are the training times for the second epoch of `examples/train.py` with
arguments as per the comments with `--smoke`, e.g. for the torch cuda (serial) example,
the comment is `# KERAS_BACKEND=torch --original_cuda`, so the result was from

```bash
KERAS_BACKEND=torch python examples/train.py --smoke --original_cuda
```
"""
import matplotlib.pyplot as plt

# Data
runs = [
    "torch cuda (serial)",  # KERAS_BACKEND=torch --original_cuda
    "torch triton (serial)",  # KERAS_BACKEND=torch
    "torch triton (parallel)",  # KERAS_BACKEND=torch --parallel_wkv
    "jax (serial)",  # KERAS_BACKEND=jax
    "jax (parallel)",  # KERAS_BACKEND=jax --parallel_wkv
    # "tensorflow (serial)",  # KERAS_BACKEND=tensorflow
    "tensorflow (parallel)",  # KERAS_BACKEND=tensorflow --parallel_wkv
]
times = [
    231,
    399,
    351,
    394,
    159,
    # 2000,
    194,
]

colors = [
    "green",
    "orange",
    "orange",
    "blue",
    "blue",
    # "black",
    "black",
]

# Create a bar chart
plt.bar(runs, times, color=colors)

# Labeling the axes
plt.ylabel("Train step time (ms)")
# plt.xlabel("Run")

# Title of the plot
# plt.title("Fruit Sales")
plt.xticks(rotation=20)
# plt.yscale("log")

# Display the plot
plt.show()
