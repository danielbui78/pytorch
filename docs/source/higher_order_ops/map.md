(map)=

# Control Flow - Map

`torch.map` is a structured control flow operator that applies a function over the leading dimension of input tensors.
It can logically be seen as implemented as follows:

```python
def map(
    f: Callable[[PyTree, ...], PyTree],
    xs: Union[PyTree, torch.Tensor],
    *args,
):
    out = []
    for idx in range(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    return torch.stack(out)
```

Its unique power lies in its ability to express **batched operations with a custom function**: it lowers to a map
operator (`torch.ops.higher_order.map_impl`), which preserves the mapping function and the structure of the inputs.
This allows for efficient parallel execution and optimization of repeated operations across a batch dimension.

```{warning}
`torch._higher_order_ops.map` is a prototype feature in PyTorch. It currently does not support autograd and you may
run into miscompiles. Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses map to apply a function over a batch:

```python
import torch
from torch._higher_order_ops import map

def f(x):
    return x.sin() + x.cos()

xs = torch.randn(3, 4, 5)  # batch of 3 tensors, each 4x5
# Applies f to each of the 3 slices
result = map(f, xs)  # returns tensor of shape [3, 4, 5]
```

You can also use map with nested inputs and additional arguments:

```python
import torch
from torch._higher_order_ops import map

def f(xs, const):
    return xs[0] + xs[1] + const

xs = [torch.randn(2, 3), torch.randn(2, 3)]  # two tensors to iterate over
const = torch.randn(3)  # broadcast constant
# returns a tensor of shape [2, 3]
result = map(f, xs, const)
```

We can export the model with map for further transformations and deployment:

```python
import torch
from torch._higher_order_ops import map

class MapModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def body_fn(x):
            return x.sin() + x.cos()

        return map(body_fn, xs)

mod = MapModule()
inp = torch.randn(3, 4)
ep = torch.export.export(mod, (inp,))
print(ep)
```

This gives us an exported program that preserves the map structure:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[3, 4]):
        body_graph_0 = self.body_graph_0
        map_impl: f32[3, 4] = torch.ops.higher_order.map_impl(body_graph_0, [arg0_1], [])
        return (map_impl,)

    class body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: f32[4]):
            sin: f32[4] = torch.ops.aten.sin.default(arg0_1)
            cos: f32[4] = torch.ops.aten.cos.default(arg0_1)
            add: f32[4] = torch.ops.aten.add.Tensor(sin, cos)
            return add
```

Notice that `torch.map` is lowered to `torch.ops.higher_order.map_impl`, and the body function becomes a
sub-graph attribute of the top-level graph module.

## Invariants of torch.ops.higher_order.map_impl

There are several useful invariants for `torch.ops.higher_order.map_impl`:

- For the body function:
    - The input and output signature will be flattened tuples.
    - It is a `torch.fx.GraphModule`.
    - Closures in the original function become explicit inputs. No closures.
    - No mutations on inputs or globals are allowed.

- For mapped inputs (xs):
    - All mapped inputs must be tensors.
    - Leading dimensions must be consistent and non-zero.

- For additional arguments:
    - They are broadcast to each iteration.

- Nesting of `torch.map` in user programs becomes nested graph modules.

## Restrictions

- Mapped `xs` can only consist of tensors.
- Leading dimensions of all tensors in `xs` must be consistent and non-zero.
- The body function must not mutate inputs.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.map.map
```
