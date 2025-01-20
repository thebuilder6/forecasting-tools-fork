from forecasting_tools.ai_models.basic_model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.basic_model_interfaces.incurs_cost import (
    IncursCost,
)
from forecasting_tools.ai_models.basic_model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.basic_model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.retryable_model import (
    RetryableModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)
from forecasting_tools.ai_models.claude35sonnet import Claude35Sonnet
from forecasting_tools.ai_models.exa_searcher import ExaSearcher
from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.gpt4ovision import Gpt4oVision
from forecasting_tools.ai_models.gpto1preview import GptO1Preview
from forecasting_tools.ai_models.metaculus4o import Gpt4oMetaculusProxy
from forecasting_tools.ai_models.perplexity import Perplexity


class ModelsToTest:
    ALL_MODELS = [
        Gpt4o,
        Gpt4oMetaculusProxy,
        Gpt4oVision,
        GptO1Preview,
        # GptO1, # TODO: dependencies do not yet support this
        Claude35Sonnet,
        Perplexity,
        ExaSearcher,
        Gemini2Exp,
        Gemini2Flash,
        Gemini2FlashThinking,
    ]
    BASIC_MODEL_LIST: list[type[AiModel]] = [
        model for model in ALL_MODELS if issubclass(model, AiModel)
    ]
    RETRYABLE_LIST: list[type[RetryableModel]] = [
        model for model in ALL_MODELS if issubclass(model, RetryableModel)
    ]
    TIME_LIMITED_LIST: list[type[TimeLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TimeLimitedModel)
    ]
    REQUEST_LIMITED_LIST: list[type[RequestLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, RequestLimitedModel)
    ]
    TOKEN_LIMITED_LIST: list[type[TokenLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TokenLimitedModel)
    ]
    INCURS_COST_LIST: list[type[IncursCost]] = [
        model for model in ALL_MODELS if issubclass(model, IncursCost)
    ]
    OUTPUTS_TEXT: list[type[OutputsText]] = [
        model for model in ALL_MODELS if issubclass(model, OutputsText)
    ]
    TOKENS_INCUR_COST_LIST: list[type[TokensIncurCost]] = [
        model for model in ALL_MODELS if issubclass(model, TokensIncurCost)
    ]
