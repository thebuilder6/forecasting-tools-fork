import os
from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    TemplateBot,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gemini2flashthinking import Gemini2FlashThinking


class GeminiFlashThinkingExpBot(TemplateBot):
    # ... rest of the class code ...

class GeminiExpBot(TemplateBot):
    # ... rest of the class code ...

class GeminiFlash2Bot(TemplateBot):
    # ... rest of the class code ...