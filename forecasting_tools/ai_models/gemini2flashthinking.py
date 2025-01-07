
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

    # Replace with the actual model name when available
    MODEL_NAME: Final[str] = "gemini-2.0-flash-thinking-exp-1219"  # Hypothetical name

    # The following are placeholder values - update with actual limits from Google
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 150  # Example limit
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 30  # Example timeout
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 50000  # Example limit
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60

