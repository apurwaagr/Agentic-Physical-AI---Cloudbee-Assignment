# Part 3A — Agent Reflection

## What worked well

The agent correctly handles the full task cycle:
1. **Initial planning via LLM** — the planning prompt asks the model to reason about which tools to call and in what order. Even when Ollama is unavailable, the deterministic fallback produces a semantically correct plan.
2. **Failure detection and replanning** — when `pick("obj_003")` returns `{"success": False, "reason": "grasp_failed"}`, the agent immediately detects the failure, sends a replan prompt to the LLM, marks the object as `unrecoverable` in world state, and skips the place step.
3. **World state tracking** — `update_world_state` is called after *every* tool call, ensuring the JSON log always reflects the true physical state rather than the assumed plan state.
4. **Correct `table_clear` semantics** — the flag is `True` only if every non-table object is either `placed` or `unrecoverable`. This is intentional: a table with an unrecoverable object that was physically attempted is "as clear as possible."

## What I would improve with more time

1. **Multi-turn LLM conversation** — currently the agent sends isolated prompts. A proper implementation would maintain a conversation history so the LLM's replan response is contextualised by the full execution history.
2. **Tool schema as JSON** — passing tool descriptions in JSON (like OpenAI function calling) would allow the LLM to output structured JSON `{"tool": "pick", "args": {"object_id": "obj_002"}}` instead of free text, making parsing reliable.
3. **Retry logic** — for transient failures (e.g. network error on a real camera scan), the agent should retry up to N times before marking unrecoverable.
4. **Partial-grasp recovery** — when `grasp_failed`, a real robot might try a different grasp pose (e.g. rotate 90°). Exposing a `regrasp(object_id, pose_offset)` tool would let the LLM try an alternative strategy.

## Design decision: Ollama fallback

Using `urllib.request` (Python stdlib) to call a local Ollama instance avoids any `pip install` requirement. The deterministic fallback guarantees the script always produces a valid `execution_log.json` regardless of whether an LLM is running — useful for CI pipelines and offline demo environments.
