# Paged Q2/Q4 KV contract

The target decoder and resident MTP layer use the same logical cache format:

- 128 tokens per page;
- 128 initial sink tokens remain Q4;
- the latest 256 tokens remain Q4;
- a page becomes Q2 only after every token in it has left the recent window;
- K and V are stored together, token-major, as `[K heads, V heads]`;
- every component is a sequence of canonical little-endian Q2_B64 or Q4_B64
  blocks; Q3 does not exist.

`PagedKvCache::push` returns a `KvCacheUpdate`. CUDA, Metal, and Vulkan
dispatchers use its appended page/token coordinates and `demoted_pages` list to
update device residency without scanning or duplicating the complete cache.
`PagedKvCache::page_views` exposes immutable packed byte slices plus precision,
token range, and geometry. A backend may upload those slices directly; it may
not requantize them or retain a parallel dense K/V copy.

The scalar decoder dequantizes page views only as a correctness oracle before
calling grouped-query attention. That host materialization is not an admitted
production backend path. Production promotion requires fused page append,
Q4-to-Q2 demotion, and paged attention kernels with no hidden CPU fallback.

At the frozen Qwen geometry, all 16 target full-attention layers consume 9,216
Q2 bytes per token. Exactly 384 tokens are Q4 at a page-aligned 128K boundary,
adding 8,192 bytes per retained token. The independent MTP cache adds 576 Q2
bytes per token and 512 Q4-delta bytes per retained token. The Fold memory plan
also reserves page metadata, one possible boundary page, and one layer's
conversion scratch; see `MEMORY_PLAN_CORRECTION_V4.json`.
