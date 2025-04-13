from typing import Optional
from torchtune.datasets import samsum_dataset as biolaysumm_dataset


class Instruct:
    """Data formatter for instruction tasks."""

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    def format(self, output: str, input: Optional[str] = None, instruction: Optional[str] = None) -> str:
        return str(self._format(output, input, instruction))

    def _format(self, output: str, input: Optional[str] = None, instruction: Optional[str] = None) -> str:
        return {
            "instruction": instruction if instruction else self.instruction,
            "input": input if input else "",
            "output": output
        }


class Summary(Instruct):
    """Data formatter for summary tasks."""
    PROMPT = """
    {instruction}
    {article}
    ---
    Summary:
    {summary}
    """

    def __init__(self, instruction: Optional[str] = None) -> None:
        super().__init__(instruction if instruction else "Summarize the article:")

    def format(self, output: str, input: str, instruction: Optional[str] = None) -> str:
        return super().format(output, input, instruction)

    def get_prompt(self, output: str, input: str, instruction: Optional[str] = None) -> str:
        data = self._format(output, input, instruction)
        return self.PROMPT.format(
            instruction=data["instruction"],
            article=data["input"],
            summary=data["output"]
        )


if __name__ == '__main__':
    dataset = biolaysumm_dataset()