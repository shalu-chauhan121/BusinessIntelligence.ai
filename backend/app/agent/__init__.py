"""
The agentic investigation layer.

Everything under `app.agent` is new code built for the tool-calling agent that
replaces the fixed observe -> investigate -> contest -> act pipeline. Nothing
here talks to a language model directly; this package is the ground truth the
model's tool calls are checked against and executed on.
"""
