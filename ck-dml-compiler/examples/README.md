# Examples

Two small programs, saved after each compiler stage so the stages can be read side by side.

| File | Stage |
|---|---|
| `effects.mlir` | 3 effect queries on one model, as the emitter writes them |
| `effects.1-decomposed.mlir` | after `--dml-decompose`: 3 `dml.cate` + 3 `dml.scale` |
| `effects.2-cse.mlir` | after `--cse`: 1 `dml.cate` + 3 `dml.scale` |
| `effects.3-lowered.mlir` | after `--dml-lower-to-linalg`: standard dialects only |
| `ate.*` | the same for 2 models by 2 scenarios of average effect (shows mean pushdown) |

Regenerate one by hand:

```bash
./dev bash -c '$DML_OPT --dml-decompose --cse examples/effects.mlir'
```
