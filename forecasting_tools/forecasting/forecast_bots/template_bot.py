import logging
import os
import re
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.claude35sonnet import Claude35Sonnet
from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.perplexity import Perplexity
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


class TemplateBot(ForecastBot):
    FINAL_DECISION_LLM = (
        Gemini2FlashThinking(temperature=0.7)  # Prioritize Gemini
        if os.getenv("GOOGLE_API_KEY")
        else (
            Gpt4o(temperature=0.7)
            if os.getenv("OPENAI_API_KEY")
            else (
                Gpt4oMetaculusProxy(temperature=0.7)
                if os.getenv("METACULUS_TOKEN")
                else (
                    Claude35Sonnet(temperature=0.7)
                    if os.getenv("ANTHROPIC_API_KEY")
                    else Gpt4o(temperature=0.7)
                )
            )
        )
    )

    async def run_research(self, question: MetaculusQuestion) -> str:
        system_prompt = clean_indents(
            """
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            """
        )
        prompt = clean_indents(
            f"""
            Question:
            {question.question_text}

            {question.resolution_criteria}

            {question.fine_print}

            {question.background_info}
            """
        )
        if os.getenv("PERPLEXITY_API_KEY"):
            response = await Perplexity(system_prompt=system_prompt).invoke(
                prompt
            )
        elif os.getenv("EXA_API_KEY"):
            response = await SmartSearcher().invoke(prompt)
        else:
            logger.error(
                "No API keys for searching the web. Skipping research and setting it blank."
            )
            response = ""
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=1, min_prediction=0
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_multiple_choice_rationale(
            reasoning, question.options
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            Never use scientific notation.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_numeric_rationale(
            reasoning, question
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _extract_forecast_from_binary_rationale(
        self, rationale: str, max_prediction: float, min_prediction: float
    ) -> float:
        assert 0 <= max_prediction <= 1
        assert 0 <= min_prediction <= 1
        assert max_prediction >= min_prediction
        matches = re.findall(r"(\d+)%", rationale)
        if matches:
            # Return the last number found before a '%'
            original_number = int(matches[-1]) / 100
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            assert (
                min_prediction <= clamped_number <= max_prediction
            ), f"Clamped number {clamped_number} is not between {min_prediction} and {max_prediction}"
            return clamped_number
        else:
            raise ValueError(
                f"Could not extract prediction from response: {rationale}"
            )

    def _extract_forecast_from_multiple_choice_rationale(
        self, reasoning: str, options: list[str]
    ) -> PredictedOptionList:
        option_probabilities = []

        # Iterate through each line in the text
        for expected_option in options:
            probability_found = False
            matching_lines = []
            for line in reasoning.split("\n"):
                if expected_option in line:
                    matching_lines.append(line)

            if matching_lines:
                last_matching_line = matching_lines[-1]
                # Extract all numbers from the line
                numbers_as_string = re.findall(
                    r"-?\d+(?:,\d{3})*(?:\.\d+)?", last_matching_line
                )
                numbers_as_float = [
                    float(num.replace(",", "")) for num in numbers_as_string
                ]
                if len(numbers_as_float) >= 1:
                    last_number = numbers_as_float[-1]
                    option_probabilities.append(last_number)
                    probability_found = True

            if not probability_found:
                raise ValueError(
                    f"No probability found for option: {expected_option}"
                )

        assert len(option_probabilities) == len(
            options
        ), f"Number of option probabilities {len(option_probabilities)} does not match number of options {len(options)}"

        total_sum = sum(option_probabilities)
        decimal_list = [x / total_sum for x in option_probabilities]

        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in decimal_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        normalized_option_probabilities = normalized_list

        predicted_options: list[PredictedOption] = []
        for i in range(len(options)):
            predicted_options.append(
                PredictedOption(
                    option_name=options[i],
                    probability=normalized_option_probabilities[i],
                )
            )

        return PredictedOptionList(predicted_options=predicted_options)

    def _extract_forecast_from_numeric_rationale(
        self, reasoning: str, question: NumericQuestion
    ) -> NumericDistribution:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in reasoning.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        percentiles = [
            Percentile(
                value=value,
                percentile=percentile / 100,
            )
            for percentile, value in results
        ]

        if not percentiles:
            raise ValueError(
                f"Could not extract prediction from response: {reasoning}"
            )

        return NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )
