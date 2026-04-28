"""Module I inference -- backend protocol, output parser, smoke-test harness.

Per master directive section 0.5.1.A bullet 13. The smoke-test harness
exercises the full Module I pipeline (persona system prompt + user
prompt + LLM generate + JSON parse + Pydantic validation) and asserts
the record set passes the AlphaLedgerRecord schema. It is designed for
*plumbing validation* -- output is tagged so it cannot be confused with
research output.

A NullBackend is provided so the harness runs in CI without any LLM
dependency. Real backends (vLLM, llama-cpp-python) are expected to
implement the InferenceBackend protocol and plug in at the call site.
"""
