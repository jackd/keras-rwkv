import typing as tp
import tree
import keras_core as keras

Node = tp.Any


class InconsistentCacheException(Exception):
    pass


def _run_through_node_graph(
    func: keras.Function, inputs, node_fn: tp.Callable[[Node], tp.Callable]
):
    """
    Execute the graph with a custom node function.

    At each node we compute outputs via
    `node_fn(node)(*args, **kwargs)`.

    This is almost identical to `_run_through_graph` but allows custom hooks at the
    node level. In fact, we could re-implement `_run_through_graph` as

    ```python
    def _run_through_graph(self, inputs, operation_fn):
        return self._run_through_node_graph(
            inputs, lambda node: operation_fn(node.operation)
        )
    ```
    """
    # This implementation is based on `Functional._run_through_graph`. Note we
    # cannot use `_run_through_graph` directly because caches must be built on a
    # node level rather than an operation level
    inputs = tree.flatten(inputs)

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}
    for x, y in zip(func.inputs, inputs):
        tensor_dict[id(x)] = y

    nodes_by_depth = func._nodes_by_depth  # pylint:disable=protected-access
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
            # tf.keras uses node.layer, keras_core uses node.operation
            operation = getattr(node, "operation", getattr(node, "layer", None))
            if not operation or node.is_input:
                continue  # Input tensors already exist.

            if any(id(x) not in tensor_dict for x in node.input_tensors):
                continue  # Node is not computable, try skipping.

            args, kwargs = node.arguments.fill_in(tensor_dict)

            outputs = node_fn(node)(*args, **kwargs)

            # Update tensor_dict.
            for x, y in zip(node.outputs, tree.flatten(outputs)):
                tensor_dict[id(x)] = y

    output_tensors = []
    for x in func.outputs:
        output_tensors.append(tensor_dict[id(x)])

    hidden_state = tree.unflatten_as(
        func._outputs_struct, output_tensors  # pylint:disable=protected-access
    )
    return hidden_state


def call_and_create_cache(func, inputs, *, current_index, max_length, mask=None):
    """
    Call the given function and get standard `call` output plus associated cache.

    This is designed to be called once as the starting point for efficient generation
    from a `keras.Function`.

    If `func` implements `call_and_create_cache` itself this function redirects there.
    Otherwise, `func` must be a `keras.Function` (functional `keras.Model`s are
    `Function`s) and this function crawls `func`s node graph and call each operation
    either:
        - using `operation.call_and_create_cache` if it's implemented;
        - using `call_and_create_cache(operation)` for `keras.Function`s; or
        - using `operation.__call__` otherwise.

    If either of the first two paths are taken, the output is passed along as per the
    standard graph flow, and the returned cache is saved under `id(node)` key in the
    returned cache dictionary.

    Args:
        func: `keras.Function` or something implementing `call_and_create_cache`
        inputs: inputs to `func`
        current_index: index of the token for which predictions/cache should be made.
            To use the returned cache with `call_with_cache`, use `current_index + 1`
            in `call_with_cache`.
        max_length: maximum length that will be generated to.
        mask: annoyingly necessary

    Returns:
        (func_output, cache)
    """
    if hasattr(func, "call_and_create_cache"):
        return func.call_and_create_cache(
            inputs, current_index=current_index, max_length=max_length, mask=mask
        )
    if not isinstance(func, keras.Function):
        raise ValueError(
            "func must be a keras.Function or implement `call_and_create_cache`, got "
            f"{func}"
        )

    cache = {}

    def node_fn(node):
        operation = node.operation

        if hasattr(operation, "call_and_create_cache"):
            assert hasattr(operation, "call_with_cache"), operation

            def call_cachable_operation(*args, **kwargs):
                output, cache[id(node)] = operation.call_and_create_cache(
                    *args,
                    current_index=current_index,
                    max_length=max_length,
                    **kwargs,
                )
                return output

            return call_cachable_operation

        if isinstance(operation, keras.Function):

            def call_function(*args, **kwargs):
                output, op_cache = call_and_create_cache(
                    operation,
                    *args,
                    current_index=current_index,
                    max_length=max_length,
                    **kwargs,
                )
                if op_cache:
                    cache[id(node)] = op_cache
                return output

            return call_function

        if isinstance(operation, keras.Layer):

            def call_layer(*args, **kwargs):
                output, op_cache = _call_and_create_cache_from_call(
                    node,
                    *args,
                    current_index=current_index,
                    max_length=max_length,
                    **kwargs,
                )
                if op_cache:
                    cache[id(node)] = op_cache
                return output

            return call_layer

        # default
        return operation

    output = _run_through_node_graph(func, inputs, node_fn)
    return output, cache


def _validate_cache_structure(operation, input_cache, output_cache):
    try:
        tree.assert_same_structure(input_cache, output_cache)

        def check(cache_in, cache_out):
            assert cache_in.shape == cache_out.shape, (cache_in.shape, cache_out.shape)
            assert cache_in.dtype == cache_out.dtype, (cache_in.dtype, cache_out.dtype)

        tree.map_structure(check, input_cache, output_cache)

    except (AssertionError, ValueError) as e:
        raise InconsistentCacheException(
            f"Inconsistent cache update from operation {operation}"
        ) from e


def call_with_cache(
    func, inputs, *, current_index, cache: tp.Mapping[int, tp.Any], mask=None
):
    """
    Call the given function with a cache.

    This is designed to be called recursively following `call_and_create_cache`.

    If `func` implements `call_with_cache` this function returns the output of that one.
    Otherwise, `func` must be a `keras.Function` (which includes functional
    `keras.Model`s). In this case we crawl `func`'s `Node` graph and for each node
    we with an id in `cache` we call either `operation.call_with_cache` if it is
    implemented, or `call_with_cache(operation, ...)` otherwise.

    The `cache` is generally generated by `call_and_create_cache`.

    Args:
        func: something implementing `func.call_with_cache` or a `keras.Function`.
        inputs: inputs to func's standard `call`.
        current_index: index of the token most recently generated.
        cache: mapping from node `id`s to a structure of cached objects.
        mask: annoyingly necessary.

    Returns:
        (func_output, updated_cache)
    """
    if hasattr(func, "call_with_cache"):
        return func.call_with_cache(
            inputs, current_index=current_index, cache=cache, mask=mask
        )
    if not isinstance(func, keras.Function):
        raise ValueError(
            f"func must be a keras.Function of implement `call_with_cache`, got {func}"
        )

    updated_cache = {}

    def node_fn(node):
        operation = node.operation
        node_id = id(node)

        if node_id not in cache:
            return operation

        op_cache = cache[node_id]
        if hasattr(operation, "call_with_cache"):

            def call_custom_implementation(*args, **kwargs):
                output, cache_update = operation.call_with_cache(
                    *args,
                    cache=op_cache,
                    current_index=current_index,
                    **kwargs,
                )
                _validate_cache_structure(operation, op_cache, cache_update)
                updated_cache[node_id] = cache_update
                return output

            return call_custom_implementation

        if isinstance(operation, keras.Function):

            def call_function_fallback(*args, **kwargs):
                output, op_updated_cache = call_with_cache(
                    operation,
                    *args,
                    current_index=current_index,
                    cache=op_cache,
                    **kwargs,
                )
                _validate_cache_structure(operation, op_cache, op_updated_cache)
                updated_cache[node_id] = op_updated_cache
                return output

            return call_function_fallback

        if isinstance(operation, keras.Layer):

            def call_layer_fallback(*args, **kwargs):
                output, op_updated_cache = _call_with_cache_from_call(
                    node,
                    *args,
                    current_index=current_index,
                    cache=op_cache,
                    **kwargs,
                )
                _validate_cache_structure(operation, op_cache, op_updated_cache)
                updated_cache[node_id] = op_updated_cache
                return output

            return call_layer_fallback

        raise ValueError(
            f"operation {operation} must either implement `call_with_cache`, be a "
            "`keras.Function` or be a `keras.Operation` but is none of these."
        )

    output = _run_through_node_graph(func, inputs, node_fn)
    _validate_cache_structure(func, cache, updated_cache)
    return output, updated_cache


def _call_to_func(operation: keras.Operation, args, kwargs, static_kwargs, name):
    inputs = (input_args, input_kwargs) = tree.map_structure(
        lambda x: keras.KerasTensor(x.shape, x.dtype),
        (args, kwargs),
    )
    outputs = operation.call(*input_args, **input_kwargs, **static_kwargs)
    return keras.Function(inputs, outputs, name=name)


def _call_and_create_cache_from_call(node, *args, current_index, max_length, **kwargs):
    operation: keras.Operation = node.operation
    assert isinstance(operation, keras.Operation), operation
    func = getattr(node, "_call_func", None)
    static_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, "shape")}
    kwargs = {k: v for k, v in kwargs.items() if k not in static_kwargs}
    # print(operation, args, kwargs)
    if func is None:
        func = _call_to_func(
            operation,
            args,
            kwargs,
            static_kwargs,
            name=f"{operation.name}_call_and_create",
        )
        # we save the `func` for use with `call_with_cache_from_call` because the cache
        # uses `id(node)` keys. If we regenerate `func` then the new nodes will have
        # different ids.
        node._call_func = func
    return call_and_create_cache(
        func, (args, kwargs), current_index=current_index, max_length=max_length
    )


def _call_with_cache_from_call(node, *args, current_index, cache, **kwargs):
    operation: keras.Operation = node.operation
    assert isinstance(operation, keras.Operation), operation

    func = node._call_func
    static_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, "shape")}
    kwargs = {k: v for k, v in kwargs.items() if k not in static_kwargs}
    return call_with_cache(
        func, (args, kwargs), current_index=current_index, cache=cache
    )
