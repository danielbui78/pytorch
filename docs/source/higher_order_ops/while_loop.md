(while_loop)=

# Control Flow - While Loop

`torch.while_loop` is a structured control flow operator that runs a body function while a condition is true.
It can logically be seen as implemented as follows:

```python
def while_loop(
    cond_fn: Callable[..., bool],
    body_fn: Callable[..., tuple],
    carried_inputs: tuple,
):
    val = carried_inputs
    while cond_fn(*val):
        val = body_fn(*val)
    return val
```

Its unique power lies in its ability to express **data-dependent iteration**: it lowers to a while_loop
operator (`torch.ops.higher_order.while_loop`), which preserves the condition function, body function,
and the loop-carried state. This enables models with dynamic iteration counts that depend on runtime
values.

```{warning}
`torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types
and doesn't support training currently. Please look forward to a more stable implementation in a future
version of PyTorch. Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is a basic example that uses while_loop to iterate until a condition is met:

```python
import torch
from torch._higher_order_ops import while_loop

def cond_fn(iter_count, x):
    return iter_count.sum() < 10

def body_fn(iter_count, x):
    return iter_count + 1, x.sin()

init_iter = torch.zeros(1)
init_x = torch.randn(3, 4)

final_iter, final_x = while_loop(cond_fn, body_fn, (init_iter, init_x))
```

Here is an example with integer loop counters:

```python
import torch
from torch._higher_order_ops import while_loop

def cond_fn(int_iter, x):
    return 2 * int_iter < x.shape[0]

def body_fn(int_iter, x):
    return int_iter + 1, x + int_iter

# When passing an integer as carry, the return will also be an integer
# with unknown value (since iteration count is data-dependent)
final_iter, final_x = while_loop(cond_fn, body_fn, (0, torch.randn(3, 4)))
```

You can use while_loop with nested data structures:

```python
import torch
from torch._higher_order_ops import while_loop

def cond_fn(state):
    return state['count'] < 5

def body_fn(state):
    return {
        'count': state['count'] + 1,
        'value': state['value'] * 2,
    }

init_state = {
    'count': torch.tensor(0),
    'value': torch.tensor(1.0),
}

final_state = while_loop(cond_fn, body_fn, (init_state,))
```

We can export the model with while_loop for further transformations and deployment:

```python
import torch
from torch._higher_order_ops import while_loop

class WhileLoopModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def cond_fn(i, val):
            return i < 10

        def body_fn(i, val):
            return i + 1, val.sin()

        init_i = torch.tensor(0)
        _, result = while_loop(cond_fn, body_fn, (init_i, x))
        return result

mod = WhileLoopModule()
inp = torch.randn(3, 4)
ep = torch.export.export(mod, (inp,))
print(ep)
```

This gives us an exported program that preserves the while_loop structure:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[3, 4]):
        _tensor_constant0: i64[] = self._tensor_constant0
        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(
            while_loop_cond_graph_0,
            while_loop_body_graph_0,
            [_tensor_constant0, arg0_1],
            []
        )
        getitem_1: f32[3, 4] = while_loop[1]
        return (getitem_1,)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: i64[], arg1_1: f32[3, 4]):
            lt: b8[] = torch.ops.aten.lt.Scalar(arg0_1, 10)
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: i64[], arg1_1: f32[3, 4]):
            add: i64[] = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: f32[3, 4] = torch.ops.aten.sin.default(arg1_1)
            return (add, sin)
```

Notice that `torch.while_loop` is lowered to `torch.ops.higher_order.while_loop`, and both the condition
and body functions become sub-graph attributes of the top-level graph module.

## Using while_loop with nn.Module

You can access module parameters and buffers inside the condition and body functions:

```python
import torch
from torch._higher_order_ops import while_loop

class SimpleWithLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer('threshold', torch.tensor(5))

    def forward(self, iter_count, x):
        def cond_fn(it, val):
            return it < self.threshold

        def body_fn(it, val):
            return it + 1, self.linear(val)

        return while_loop(cond_fn, body_fn, (iter_count, x))

mod = SimpleWithLinear()
result = mod(torch.tensor(0), torch.randn(3, 2))
```

## Invariants of torch.ops.higher_order.while_loop

There are several useful invariants for `torch.ops.higher_order.while_loop`:

- For the condition function:
    - Returns a boolean scalar tensor or Python boolean.
    - Takes the same inputs as the body function.
    - It is a `torch.fx.GraphModule` after tracing.

- For the body function:
    - Returns a tuple with the same structure and metadata as the inputs.
    - Takes carried_inputs and additional_inputs.
    - It is a `torch.fx.GraphModule` after tracing.
    - Closures become additional_inputs.

- For carried_inputs:
    - Can be nested dict/list/tuple of tensors or integers.
    - When passing an integer as carry, the return will be an integer with unknown value.

- For outputs:
    - Returns the same structure as carried_inputs.
    - Tensors maintain the same shape and dtype as inputs.

## Restrictions

- `body_fn` must return tensors or integers with the same metadata (shape, dtype) as inputs.

- `body_fn` and `cond_fn` must not in-place mutate the `carried_inputs`. A clone before mutation is required.

- `body_fn` and `cond_fn` must not mutate Python variables (e.g., list/dict) created outside the function.

- `body_fn` and `cond_fn`'s output cannot alias any of the inputs. A clone is required.

```{warning}
**Temporal Limitations:**

`while_loop` only supports **inference** right now. Autograd support will be added in the future.
```

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.while_loop.while_loop
```
