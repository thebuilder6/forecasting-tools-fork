from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)
from forecasting_tools.ai_models.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)


class BasicLlm(Gemini2FlashThinking):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(Gemini2FlashThinking):
    pass


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
