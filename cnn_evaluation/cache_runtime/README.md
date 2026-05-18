First-stage cache runtime for `SHADE + CIDS`.

Current scope:

- host-side sample cache
- simple LRU eviction
- batch bridge that checks cache first and falls back to CIDS

Not included yet:

- distributed cache
- importance-aware admission
- dedicated registered buffer pool
- integration into the existing training scripts

Files:

- `sample_cache.py`
  simple host sample cache
- `sample_importance.py`
  minimal per-sample importance tracker based on EMA loss
- `cids_cached_loader.py`
  batch-level bridge:
  cache hit -> host cache
  cache miss -> CIDS sync / registered

Typical usage:

```python
from cache_runtime.sample_cache import LRUSampleCache
from cache_runtime.cids_cached_loader import CachedCIDSBatchLoader

sample_cache = LRUSampleCache(capacity=4096, pin_memory=True)
cached_loader = CachedCIDSBatchLoader(
    cids_loader=train_cids,
    sample_cache=sample_cache,
    io_mode="registered",
)

for raw_batch in dataloader:
    batch = cached_loader.fetch_batch(raw_batch)
    images, labels = batch
```
