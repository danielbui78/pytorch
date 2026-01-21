(scan)=

# Control Flow - Scan

`torch.scan` is a structured control flow operator that performs an inclusive scan with a combine function.
It is commonly used for cumulative operations like cumsum, cumprod, or more general recurrences.
It can logically be seen as implemented as follows:

```python
def scan(
    combine_fn: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
    init: PyTree,
    xs: PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> tuple[PyTree, PyTree]:
    carry = init
    ys = []
    for i in range(xs.size(dim)):
        x_slice = xs.select(dim, i)
        carry, y = combine_fn(carry, x_slice)
        ys.append(y)
    return carry, torch.stack(ys)
```

Its unique power lies in its ability to express **sequential dependencies with state**: it lowers to a scan
operator (`torch.ops.higher_order.scan`), which preserves the combine function and the structure of carries
and outputs. This enables efficient lowering to optimized implementations while maintaining the sequential
semantics.

```{warning}
`torch.scan` is a prototype feature in PyTorch. You may run into miscompiles.
Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses scan to compute a cumulative sum:

```python
import torch
from torch._higher_order_ops import scan

def add(carry: torch.Tensor, x: torch.Tensor):
    next_carry = carry + x
    y = next_carry.clone()  # clone to avoid output-output aliasing
    return next_carry, y

init = torch.zeros(1)
xs = torch.arange(5, dtype=torch.float32)

final_carry, cumsum = scan(add, init=init, xs=xs)
# final_carry = tensor([10.])
# cumsum = tensor([[0.], [1.], [3.], [6.], [10.]])
```

Here is an example where the carry and output have different structures:

```python
import torch
from torch._higher_order_ops import scan

def combine(carry: torch.Tensor, x: torch.Tensor):
    # carry is a running product, output is running sum
    next_carry = carry * x
    y = carry + x
    return next_carry, y

init = torch.ones(1, dtype=torch.int64)
xs = torch.arange(1, 5, dtype=torch.int64)

final_carry, result = scan(combine, init=init, xs=xs)
# final_carry = tensor([24])  # 1 * 2 * 3 * 4 = 24
# result = tensor([[2], [3], [5], [10]])  # cumulative sums at each step
```

You can also scan over a specific dimension and in reverse:

```python
import torch
from torch._higher_order_ops import scan

def add(carry, x):
    y = carry + x
    return y, y.clone()

init = torch.zeros(3)
xs = torch.randn(5, 3)  # scan over dim=0

# Forward scan
final_carry, cumsum = scan(add, init=init, xs=xs, dim=0, reverse=False)

# Reverse scan (processes elements from last to first)
final_carry_rev, cumsum_rev = scan(add, init=init, xs=xs, dim=0, reverse=True)
```

We can export the model with scan for further transformations and deployment:

```python
import torch
from torch._higher_order_ops import scan

class ScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def combine_fn(carry, x):
            next_carry = carry + x
            return next_carry, next_carry.clone()

        init = torch.zeros_like(xs[0])
        return scan(combine_fn, init=init, xs=xs)

mod = ScanModule()
inp = torch.randn(5, 3)
ep = torch.export.export(mod, (inp,))
print(ep)
```

This gives us an exported program that preserves the scan structure:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[5, 3]):
        slice_1: f32[3] = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 1)
        zeros_like: f32[3] = torch.ops.aten.zeros_like.default(slice_1)
        scan_combine_graph_0 = self.scan_combine_graph_0
        scan = torch.ops.higher_order.scan(scan_combine_graph_0, [zeros_like], [arg0_1], [])
        return scan

    class scan_combine_graph_0(torch.nn.Module):
        def forward(self, arg0_1: f32[3], arg1_1: f32[3]):
            add: f32[3] = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
            clone: f32[3] = torch.ops.aten.clone.default(add)
            return (add, clone)
```

## Invariants of torch.ops.higher_order.scan

There are several useful invariants for `torch.ops.higher_order.scan`:

- For the combine function:
    - It takes two arguments: the current carry and a slice of the input.
    - It returns two values: the next carry and an output for this step.
    - It is a `torch.fx.GraphModule` after tracing.
    - Must be pure: no lifted arguments are supported and no side effects.
    - No mutations on inputs are allowed.
    - No aliasing between input-input, input-output, or output-output.

- For init (initial carry):
    - Must have the same pytree structure as the first output element (carry) of `combine_fn`.
    - All leaves must be tensors.

- For xs (inputs to scan over):
    - All leaves must be tensors.
    - Each tensor must have at least `dim + 1` dimensions.
    - The scan dimension must have size > 0.

- For outputs:
    - Returns a tuple of `(final_carry, stacked_outputs)`.
    - `final_carry` has the same structure and shapes as `init`.
    - `stacked_outputs` has the outputs from each iteration stacked along dimension 0.

## Restrictions

- The `combine_fn` must not alias between input-input, input-output, or output-output. As a workaround, clone the output to avoid aliasing.

- The `combine_fn` must not mutate inputs. Mutation support for inference will be added in the future.

- The `combine_fn`'s init carry must match the next_carry in pytree structure and tensor metadata.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.scan.scan
```
