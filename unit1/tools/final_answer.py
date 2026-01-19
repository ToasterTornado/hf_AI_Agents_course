from typing import Any, Optional
from smolagents.tools import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # Ensure the answer is treated as a string and newlines are interpreted correctly
        formatted_answer = str(answer).replace('\\n', '\n')
        return formatted_answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
