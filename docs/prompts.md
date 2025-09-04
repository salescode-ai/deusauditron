# Prompts & Templates

Prompt files live in `deusauditron/prompts/`.

## Evaluation Prompts

- `turn.evaulation.prompt` — turn-level evaluation
- `node.evaulation.prompt` — node-level evaluation
- `conversation.evaulation.prompt` — flow-level evaluation
- `intent.evaulation.prompt` — intent-level evaluation

## Auto-refinement Prompts

- `auto_refine_turn_node.prompt`
- `auto_refine_intent.prompt`
- `auto_refine_flow.prompt`
- `auto_refine_response_conversion.prompt`
- `auto_refine_flow_response_conversion.prompt`
- `auto_refine_intent_response_conversion.prompt`
- `llm_response_conversion.prompt`
- `default_intent.prompt`

## Access in Code

Prompts are loaded by `Trinity.get_*_prompt()` helpers in `deusauditron/util/helper.py`.
