from typing import Final

from forecasting_tools.ai_models.model_archetypes.google_model import (
    GoogleTextToTextModel,
)


class Gemini2FlashThinking(GoogleTextToTextModel):
    """
    Represents the Gemini 2 Flash Thinking model for text generation.

    This class inherits from GoogleTextToTextModel and sets default parameters
    specific to the Gemini 2 Flash Thinking model.
    """

    MODEL_NAME: Final[str] = "gemini-exp-1206"
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 30
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 50000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
