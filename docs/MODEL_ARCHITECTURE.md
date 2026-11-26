# Model Architecture — AnkaLM Decoder-Only Transformer

## Architectural Decisions
1. **Pre-Normalization with RMSNorm:** Every attention and feed-forward sub-layer applies `RMSNorm` (`eps=1e-5`) to its input before adding the residual connection, preventing gradient explosion during early random-init training.
2. **Rotary Positional Embeddings (RoPE):** Queries and keys are rotated in 2D subspaces using $\theta = 10000.0$, encoding relative token distances directly in the attention dot product.
3. **Grouped-Query Causal Attention (GQA):** In `AnkaLM-Small`, 6 query heads share 2 KV heads (`head_dim = 64`), reducing autoregressive KV-cache memory by 66% while maintaining expressive multi-head attention via PyTorch `F.scaled_dot_product_attention`.
4. **SwiGLU Feed-Forward Network:** Uses $W_{\text{down}}(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x)$ with intermediate dimension $d_{\text{ff}} = 1024$.
5. **Weight Tying & Residual Scaling:** Input token embeddings (`8192 x 384`) are tied with the output `lm_head`. Residual output projections (`o_proj`, `down_proj`) are scaled by $1/\sqrt{2L}$ at random initialization.
