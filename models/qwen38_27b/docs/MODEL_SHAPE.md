# Frozen Qwen3.8-27B text shape

The first artifact must match the immutable base revision recorded in its
manifest. The initial shape contract is:

- vocabulary: 248,320;
- hidden size: 5,120;
- FFN size: 17,408;
- decoder layers: 64;
- pattern: linear, linear, linear, full attention;
- full-attention layers: 16;
- attention heads: 24;
- KV heads: 4;
- attention head dimension: 256;
- partial rotary dimension: 64 (25 percent of each attention head);
- RoPE theta: 10,000,000;
- RMS normalization epsilon: 0.000001;
- linear-attention layers: 48;
- linear key/value dimensions: 128/128;
- linear key/value heads: 16/48;
- one resident MTP layer;
- vision is packaged separately.

Any source revision with a different shape requires a new format/profile
revision and fresh memory proof.
