# Turkish Tokenizer Guide — AnkaLM

## Turkish Unicode & Dotted/Dotless I Handling
Standard ASCII case-folding corrupts Turkish text by mapping `I` $\rightarrow$ `i` instead of `ı` and decomposing `İ`. AnkaLM enforces NFC normalization (`preserve_turkish_unicode`) and explicit `İ -> i` / `I -> ı` mapping (`turkish_lower`, `turkish_upper`), ensuring all 14 Turkish-specific characters (`ç, Ç, ğ, Ğ, ı, I, İ, i, ö, Ö, ş, Ş, ü, Ü`) are included in the initial alphabet and roundtrip losslessly.

## BPE vs. Unigram Benchmark Results (`tokenizer/tokenizer_benchmark.json`)
- **BPE-8192 (Selected):** `8,192` vocab size | `5.211` chars/token | `3.417` tokens/word morphological fragmentation | `2,873,603.6` tokens/sec encoding speed | 100% Turkish Unicode roundtrip.
- **BPE-4096:** `4,096` vocab size | `4.568` chars/token | `4.917` tokens/word morphological fragmentation.
- **Unigram-8192:** `5,164` actual vocab size | `3.687` chars/token | `4.167` tokens/word morphological fragmentation.

## Immutable Special Tokens
- `<|pad|>`: `0`, `<|bos|>`: `1`, `<|eos|>`: `2`, `<|eot|>`: `3`, `<|system|>`: `4`, `<|user|>`: `5`, `<|assistant|>`: `6`, `<|tool|>`: `7`, `<|unk|>`: `8`.
- Vocabulary SHA-256 fingerprint (`c9642b2d37dd07c6734636700fb9cb3f5069a949cf853e38ee6c29cf964cb118`) is verified on every load.
