# prompts.py

"""
Prompt Engineering Module for Qwen-VL Tunnel Lining Anomaly Detection.

This module is responsible for dynamically constructing the final prompts sent to the
vision-language model. It centralizes all prompt-related logic, allowing for easy
modification and experimentation with different prompt strategies without changing
the main execution script.

Key functionalities:
- Defines the placeholder for images (`IMAGE_TOKEN`).
- Builds the text block for few-shot examples.
- Assembles the final prompt from a template and data.
"""

from typing import List, Tuple

# This is the special token that the Llama-factory framework uses to mark
# the position where an image should be inserted in the prompt.
IMAGE_TOKEN = "<image>"


def build_few_shot_examples_text(example_images: List[Tuple[str, str]]) -> str:
    """
    Constructs the multi-example text block for a few-shot prompt.

    This function iterates through a list of example images and their labels,
    creating a formatted string that serves as the "learning" section for the model.
    The analysis text for "异常" (abnormal) and "正常" (normal) is hardcoded here
    to ensure consistency in the examples shown to the model.

    Args:
        example_images: A list of tuples, where each tuple contains:
                        - str: The file path to the example image (not used here, but part of the data structure).
                        - str: The label ('p' for positive/abnormal, 'n' for negative/normal).

    Returns:
        A formatted string containing all few-shot examples, ready to be
        inserted into a larger prompt template.
    """
    examples_text = ""
    for i, (img_path, label) in enumerate(example_images):
        # Determine the status string based on the label
        status = "异常" if label == "p" else "正常"
        
        # Assemble the text for one example
        examples_text += f"示例 {i+1} ({status}):\n"
        examples_text += f"{IMAGE_TOKEN}\n"  # Placeholder for the i-th image
        if status == "异常":
            examples_text += "【判断】: 异常\n【分析】: 图片中存在明显的结构性缺陷。\n\n"
        else:
            examples_text += "【判断】: 正常\n【分析】: 衬砌板表面完整，无明显缺陷。\n\n"
            
    return examples_text


def get_final_prompt(prompt_template: str, few_shot_examples: List[Tuple[str, str]] = None) -> str:
    """
    Dynamically assembles the final prompt from a template and optional few-shot data.

    This is the main function used by the execution script (`run_inference.py`). 
    It takes a prompt template (read from config.yaml) and fills in the necessary placeholders.

    Args:
        prompt_template: The raw prompt string from the config file, which may contain 
                         placeholders like `{few_shot_examples}` and `{image_token}`.
        few_shot_examples: An optional list of few-shot examples. This is required if the
                           template contains the `{few_shot_examples}` placeholder.

    Returns:
        The final, complete prompt string ready to be sent to the model.

    Raises:
        ValueError: If the template requires few-shot examples but none are provided.
    """
    # Check if the template is a few-shot template by looking for the placeholder
    if "{few_shot_examples}" in prompt_template:
        if not few_shot_examples:
            raise ValueError("Prompt模板需要few-shot示例，但未提供 (few_shot_examples is None or empty).")
        
        # First, build the text block for all the examples
        examples_text = build_few_shot_examples_text(few_shot_examples)
        
        # Then, insert the examples block and the final image token placeholder
        # into the main template.
        prompt = prompt_template.format(
            few_shot_examples=examples_text, 
            image_token=IMAGE_TOKEN
        )
    else:
        # For zero-shot or bbox prompts, they typically only need the final image token.
        # The .format() method will safely ignore extra placeholders if they don't exist.
        prompt = prompt_template.format(image_token=IMAGE_TOKEN)
        
    return prompt

