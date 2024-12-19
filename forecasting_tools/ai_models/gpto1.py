from typing import Any, Final

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_archetypes.openai_text_model import (
    OpenAiTextToTextModel,
)


class GptO1(OpenAiTextToTextModel):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: Final[str] = "o1-preview"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 8_000
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 120
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 2_000_000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60

    def __init__(
        self,
        *args: Any,
        temperature: float = 1,
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        assert (
            system_prompt is None
        ), "GptO1Preview does not support system prompts"
        assert (
            temperature == 1
        ), f"GptO1Preview must have temperature 1, but {temperature} was given."
        super().__init__(*args, temperature=temperature, **kwargs)

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        response = (
            super()._get_mock_return_for_direct_call_to_model_using_cheap_input()
        )
        response.total_tokens_used += 269  # Add reasoning tokens
        return response
