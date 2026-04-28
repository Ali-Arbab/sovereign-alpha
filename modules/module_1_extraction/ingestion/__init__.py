"""Document ingestion adapters for the Module I corpus.

Per master directive section 3.1 and 0.5.1.A. One module per source:
- edgar: SEC EDGAR (10-K, 10-Q, 8-K)
- fomc: Federal Reserve FOMC statements + minutes  (next commit)
- bls:  Bureau of Labor Statistics releases         (next commit)
- gdelt: GDELT global news corpus (open substitute) (next commit)

Each adapter exposes a thin client object plus pure-function URL/parsing
helpers, so the network surface is mockable and the parsing surface is
testable in isolation.
"""
