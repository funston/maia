# tinyserve

A small batched inference engine for decoder-only transformers. It exists to
exercise the serving path (prefill, incremental decode, KV caching, dynamic
batch shrinking) without the weight of a real model.

## Layout

| Path | What |
| --- | --- |
| `tinyserve/config.py` | `ModelConfig` |
| `tinyserve/kv_cache.py` | Preallocated per-layer key/value cache |
| `tinyserve/model.py` | Transformer with a cached and an uncached forward path |
| `tinyserve/engine.py` | `Engine.generate`: pad, prefill, greedy decode loop |
| `bench/bench_decode.py` | Decode throughput benchmark |
| `issues/` | Open issues |

## How generation works

1. Prompts are right-padded to the longest prompt in the batch.
2. `prefill` runs the whole padded batch once and stores every layer's keys
   and values in the `KVCache`.
3. `decode_step` feeds one new token per sequence and attends over the cache.
4. A sequence that emits EOS is dropped from the batch; the rest keep going.

`TinyTransformer.forward` is the uncached reference path: it recomputes
everything from scratch and is what the cached path is expected to match.

## Development

```
pip install -e ".[dev]"
pytest -q
python bench/bench_decode.py --help
```
