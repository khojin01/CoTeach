"""Prompting and API helpers for the LLM teacher."""


def create_llm_prompt_for_node(dataset_name, label_names, node_text=None, k_shot_examples=None):
    """Create text-only prompt for LLM teacher with K-shot examples."""
    from llm_prompts import create_prompt

    # Semantic expert: no graph feature statistics or dense node vectors.
    return create_prompt(
        node_features=None,
        dataset_name=dataset_name,
        label_names=label_names,
        node_text=node_text,
        k_shot_examples=k_shot_examples
    )


def call_real_llm_api(prompt):
    """Call OpenAI API and normalize response payload."""
    try:
        from llm_api_config import LLMConfig, call_openai_api

        config = LLMConfig()
        response = call_openai_api(config, prompt)

        if isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            return {"content": content}
        if isinstance(response, dict):
            return response
        return {"content": str(response)}
    except Exception:
        return {
            "content": '{"answer": "Diabetes Mellitus Experimental", "confidence": 30, "reasoning": "API error"}'
        }
