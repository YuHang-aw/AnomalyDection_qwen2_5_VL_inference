# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

def _render_few_shot_text(examples: List[Tuple[str, str]]) -> str:
    """
    将 few-shot 的 (path,label) 列表转为简短文字占位，提示模型“上方有示例图像”。
    label 取值：'n'/'p'/'unknown'
    """
    lines = []
    for i, (_, lb) in enumerate(examples, 1):
        if lb == "p":
            hint = "（参考判定：异常）"
        elif lb == "n":
            hint = "（参考判定：正常）"
        else:
            hint = ""
        lines.append(f"[示例{i}] 图像见上{hint}")
    return "\n".join(lines) if lines else "（示例图像见上）"

def build_text_prompt(prompt_template: str,
                      few_shot_examples: Optional[List[Tuple[str, str]]] = None) -> str:
    """
    用模板 + 可选 few-shot 文本块生成最终文本提示词。
    模板里如包含 {few_shot_examples} 占位符则会被替换。
    """
    if not prompt_template:
        return ""
    if "{few_shot_examples}" in prompt_template:
        ex_text = _render_few_shot_text(few_shot_examples or [])
        return prompt_template.replace("{few_shot_examples}", ex_text)
    return prompt_template
