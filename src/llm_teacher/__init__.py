from llm_api_config import call_real_llm_api
from llm_prompts import create_prompt


def create_llm_prompt_for_node(dataset_name, label_names, node_text=None):
    return create_prompt(
        node_features=None,
        dataset_name=dataset_name,
        label_names=label_names,
        node_text=node_text,
    )

__all__ = [
    "create_llm_prompt_for_node",
    "call_real_llm_api",
]
