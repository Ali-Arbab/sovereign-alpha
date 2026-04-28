"""Module I tokenization pipeline -- HTML clean -> chunk -> cached tokenize.

Per master directive section 0.5.1.A and 6.3. Tokenization is the largest
deterministic cost in the inference loop and must be content-addressed so
a re-run hits the cache. The cache key is `hash(text + tokenizer_version)`,
so a tokenizer upgrade invalidates only the tokenization stage -- not
ingestion, not inference, not the Alpha Ledger.

The Tokenizer protocol decouples this module from any specific backend
(tiktoken / sentencepiece / HF tokenizers / llama.cpp tokenize); a stub
WhitespaceTokenizer is provided for the bootstrap phase plumbing.
"""
