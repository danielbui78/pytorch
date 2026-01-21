(higher_order_ops)=

# Higher Order Operators

Higher Order Operators (HOPs) are structured control flow operators in PyTorch that enable expressing
complex control flow patterns in a way that is compatible with `torch.compile` and `torch.export`.
Unlike regular Python control flow, these operators preserve their semantics through compilation and
export, enabling data-dependent control flow in traced programs.

```{warning}
Higher Order Operators are prototype features in PyTorch. They may have limited support for certain
input/output types and some may not fully support training. Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Why Use Higher Order Operators?

When you use standard Python control flow (like `if`, `for`, `while`) in code that gets traced by
`torch.compile` or `torch.export`, the control flow is "specialized" based on the values at trace time.
This means:

- Branches not taken during tracing are eliminated
- Loop iterations are unrolled based on the input sizes at trace time
- Data-dependent control flow cannot be expressed

Higher Order Operators solve this by explicitly representing control flow as operators that the compiler
understands, preserving the dynamic behavior in the exported/compiled program.

## Available Operators

```{toctree}
:maxdepth: 1

cond
map
scan
while_loop
```

### Quick Comparison

| Operator | Use Case | Example |
|----------|----------|---------|
| [cond](cond.md) | If-else branching based on a condition | `cond(x.sum() > 0, true_fn, false_fn, (x,))` |
| [map](map.md) | Apply a function over the leading dimension | `map(f, xs)` |
| [scan](scan.md) | Cumulative operations with carried state | `scan(combine_fn, init, xs)` |
| [while_loop](while_loop.md) | Data-dependent iteration | `while_loop(cond_fn, body_fn, (init,))` |

## Common Patterns

### Choosing the Right Operator

- **cond**: Use when you need to choose between two code paths based on a runtime condition.
- **map**: Use when you want to apply the same operation independently to each element along a dimension.
- **scan**: Use when each iteration depends on the result of the previous iteration (sequential dependencies).
- **while_loop**: Use when the number of iterations is not known at compile time and depends on runtime values.

### General Restrictions

All higher order operators share some common restrictions:

1. **No mutations**: The functions passed to HOPs generally cannot mutate their inputs. Clone tensors before mutating.

2. **No aliasing**: Outputs cannot alias inputs. Clone outputs when needed.

3. **Consistent signatures**: Input and output structures must match the expected patterns for each operator.

4. **Pure functions**: The body/combine functions should generally be pure (no side effects).
