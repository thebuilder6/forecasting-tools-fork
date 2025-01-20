![PyPI version](https://badge.fury.io/py/forecasting-tools.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/forecasting-tools.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/Dtq4JNdXnw)
[![Downloads](https://static.pepy.tech/personalized-badge/forecasting-tools?period=month&units=international_system&left_color=black&right_color=brightgreen&left_text=Monthly%20Downloads)](https://pepy.tech/project/forecasting-tools)
[![Downloads](https://static.pepy.tech/personalized-badge/forecasting-tools?period=total&units=international_system&left_color=black&right_color=blue&left_text=Total%20Downloads)](https://pepy.tech/project/forecasting-tools)


# Quick Install
Install this package with `pip install forecasting-tools`

# Overview
Demo website: https://mokoresearch.streamlit.app/

This repository contains forecasting and research tools built with Python and Streamlit. The project aims to assist users in making predictions, conducting research, and analyzing data related to hard to answer questions (especially those from Metaculus).

Here are the tools most likely to be useful to you:
- 🎯 **Forecasting Bot:** General Forecaster that integrates with the Metaculus AI benchmarking competition. You can forecast with a pre-existing bot or override the class to customize your own (without redoing all the API code, etc)
- 🔌 **Metaculus API Wrapper:** for interacting with questions and tournaments
- 📊 **Benchmarking:** Randomly sample quality questions from Metaculus and run you bot against them so you can get an early sense of how your bot is doing by comparing to the community prediction and expected log scores.
- 🔍 **Perplexity++ Smart Searcher:** A custom AI-powered internet-informed llm powered by Exa.ai and GPT. Its a better (but slightly more expensive) alternative to Perplexity.ai that is configurable, more accurate, able to decide on filters, able to link to exact paragraphs, etc.
- 🔑 **Key Factor Analysis:** Key Factors Analysis for scoring, ranking, and prioritizing important variables in forecasting questions

Here are some other cool components and features of the project:
- **Base Rate Researcher:** for calculating event probabilities (still experimental)
- **Niche List Researcher:** for analyzing very specific lists of past events or items (still experimental)
- **Fermi Estimator:** for breaking down numerical estimates (still experimental)
- **Monetary Cost Manager:** for tracking AI and API expenses

All the examples below are in a Jupyter Notebook called `README.ipynb` which you can run locally to test the package.

Join the [discord](https://discord.gg/Dtq4JNdXnw) for updates and to give feedback (btw feedback is very appreciated, even just a quick "I did/didn't decide to use tool X for reason Y, though am busy and don't have time to elaborate" is helpful to know)

Note: This package is still in an experimental phase. The goal is to keep the package API fairly stable, though no guarantees are given at this phase. There will be special effort to keep the ForecastBot and TemplateBot APIs consistent.


# Forecasting Bot Building

## Using the Preexisting Bots

The package comes with two major pre-built bots:
- **MainBot**: The more sophisticated and expensive bot that does deeper research.
- **TemplateBot**: A simpler bot that models the Metaculus templates that's cheaper, easier to start with, and faster to run.

They both have roughly the same parameters. See below on how to use the TemplateBot to make forecasts.

### Forecasting on a Tournament


```python
from forecasting_tools import TemplateBot, MetaculusApi

# Initialize the bot
bot = TemplateBot(
    research_reports_per_question=3,  # Number of separate research attempts per question
    predictions_per_research_report=5,  # Number of predictions to make per research report
    publish_reports_to_metaculus=False,  # Whether to post the forecasts to Metaculus
    folder_to_save_reports_to="logs/forecasts/",  # Where to save detailed reports (file saving environment variable must be set)
    skip_previously_forecasted_questions=False
)

TOURNAMENT_ID = MetaculusApi.CURRENT_QUARTERLY_CUP_ID
reports = await bot.forecast_on_tournament(TOURNAMENT_ID)

# Print results (if the tournament is not active, no reports will be returned)
for report in reports:
    print(f"\nQuestion: {report.question.question_text}")
    print(f"Prediction: {report.prediction}")
```

### Forecasting Outside a Tournament


```python
from forecasting_tools import TemplateBot, BinaryQuestion, QuestionState

# Initialize the bot
bot = TemplateBot(
    research_reports_per_question=3,
    predictions_per_research_report=5,
    publish_reports_to_metaculus=False,
)

# Get and forecast a specific question
question1 = MetaculusApi.get_question_by_url(
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
)
question2 = BinaryQuestion(
    question_text="Will YouTube be blocked in Russia?",
    background_info="...", # Or 'None'
    resolution_criteria="...", # Or 'None'
    fine_print="...", # Or 'None'
    id_of_post=0, # The ID and state only matters if using Metaculus API calls
    state=QuestionState.OPEN
)

reports = await bot.forecast_questions([question1, question2])

# Print results
for report in reports:
    print(f"Question: {report.question.question_text}")
    print(f"Prediction: {report.prediction}")
    shortened_explanation = report.explanation.replace('\n', ' ')[:100]
    print(f"Reasoning: {shortened_explanation}...")
```

    Question: Will humans go extinct before 2100?
    Prediction: 0.02
    Reasoning:  # SUMMARY *Question*: Will humans go extinct before 2100? *Final Prediction*: 2.0% *Total Cost*: $0...
    Question: Will YouTube be blocked in Russia?
    Prediction: 0.7
    Reasoning:  # SUMMARY *Question*: Will YouTube be blocked in Russia? *Final Prediction*: 70.0% *Total Cost*: $0...


The bot will:
1. Research the question
2. Generate multiple independent predictions
3. Combine these predictions into a final forecast
4. Save detailed research and reasoning to the specified folder
5. Optionally post the forecast to Metaculus (if `publish_reports_to_metaculus=True`)

Note: You'll need to have your environment variables set up (see the section below)

## Making your own bot for Metaculus AI Tournament

### Join the tournament through forking (quickest)
The quickest way to join the Metaculus Benchmarking Tournament (or any other tournament) is to fork this repo, enable Github workflow/actions, and then set repository secrets. Ideally this takes less than 15min, and then you have a bot in the tournament! Later you can develop locally and then merge in changes to your fork.

There is a prewritten workflow that will run the bot every 15min, pick up new questions, and forecast on them. Automation is handled in the `.github/workflows/` folder. The `hourly-run.yaml` file runs the bot every 15 min and will skip questions it has already forecasted on.

1) **Fork the repository**: Click 'fork' in the right hand corner of the repo.
2) **Set secrets**: Go to `Settings -> Secrets and variables -> Actions -> New repository secret` and set API keys/Tokens as secrets. You will want to set your METACULUS_TOKEN. This will be used to post questions to Metaculus, and access the Metaculus OpenAI proxy (you should automatically be given some credits if you have a bot account). For additional environment variables you might want, see the section below.
3) **Enable Actions**: Go to 'Actions' then click 'Enable'. Then go to the 'Hourly Run' workflow, and click 'Enable'. To test if the workflow is working, click 'Run workflow', choose the main branch, then click the green 'Run workflow' button. This will check for new questions and forecast only on ones it has not yet successfully forecast on.

The bot should just work as is at this point. You can disable the workflow by clicking `Actions > Hourly Run > Triple dots > disable workflow`

### Join the tournament using package
You can create your own custom bot through the package in your own repo (see example below). If you do this, it is recommended that you copy the `.github/workflow` workflows and customize them to call your bot so that you can run questions automatically.

### Local Development
See the 'Local Development' section later in this README.

### Customizing the Bot
Generally all you have to do to make your own bot is inherit from the TemplateBot and override any combination of the 3 forecasting methods and the 1 research method. This saves you the headache of parsing the outputs, interacting with the Metaculus API, etc. Here is an example. It may also be helpful to look at the TemplateBot code (forecasting_tools/forecasting/forecast_bots/template_bot.py) for a more complete example. If you forked, make sure to change the code in `scripts/run_forecasts_for_ai_tournament` to call your bot to take advantage of the github actions.


```python
from forecasting_tools import (
    TemplateBot,
    MetaculusQuestion,
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
    PredictedOptionList,
    NumericDistribution,
    SmartSearcher,
    Gpt4oMetaculusProxy
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

class MyCustomBot(TemplateBot):
    async def run_research(self, question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(
            num_searches_to_run=2,
            num_sites_per_search=3
        )

        prompt = clean_indents(
            f"""
            Analyze this forecasting question:
            1. Filter for recent events in the past 6 months
            2. Don't include domains from youtube.com
            3. Look for current trends and data
            4. Find historical analogies and base rates

            Question: {question.question_text}

            Background Info: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            """
        )

        report = await searcher.invoke(prompt)
        return report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = f"Please make a prediction on the following question: {question.question_text}. The last thing you write is your final answer as: 'Probability: ZZ%', 0-100"
        reasoning = await Gpt4oMetaculusProxy(temperature=0).invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=1, min_prediction=0
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        ...

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        ...

custom_bot = MyCustomBot()
question = MetaculusApi.get_question_by_url(
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
)
report = await custom_bot.forecast_question(question)
print(f"Question: {report.question.question_text}")
print(f"Prediction: {report.prediction}")
print(f"Research: {report.research[:100]}...")
```

    Question: Will humans go extinct before 2100?
    Prediction: 0.05
    Research: # RESEARCH
    To analyze the question of whether humans will go extinct before 2100, we need to consider...


## Setting Environment Variables
Whether running locally or through Github actions, you will need to set environment variables. All environment variables you might want are in `.env.template`. Generally you only need the METACULUS_TOKEN if running the Template. Having an EXA_API_KEY (see www.exa.ai) or PERPLEXITY_API_KEY (see www.perplexity.ai) is needed for searching the web. Make sure to put these variables in your `.env` file if running locally and in the Github actions secrets if running on Github actions. Remember to set 'FILE_WRITING_ALLOWED' to true if you want to save results.

# Forecasting Tools Examples

## Benchmarking
Below is an example of how to run the benchmarker


```python
from forecasting_tools import Benchmarker, TemplateBot, BenchmarkForBot

class CustomBot(TemplateBot):
    ...

# Run benchmark on multiple bots
bots = [TemplateBot(), CustomBot()]  # Add your custom bots here
benchmarker = Benchmarker(
    forecast_bots=bots,
    number_of_questions_to_use=2,  # Recommended 100+ for meaningful results
    file_path_to_save_reports="benchmarks/"
)
benchmarks: list[BenchmarkForBot] = await benchmarker.run_benchmark()

# View results
for benchmark in benchmarks[:2]:
    print(f"--------------------------------")
    print(f"Bot: {benchmark.name}")
    print(f"Score: {benchmark.average_inverse_expected_log_score}") # Lower is better
    print(f"Num reports in benchmark: {len(benchmark.forecast_reports)}")
    print(f"Time: {benchmark.time_taken_in_minutes}min")
    print(f"Cost: ${benchmark.total_cost}")
```

    --------------------------------
    Bot: Benchmark for TemplateBot
    Score: 0.771654061941104
    Num reports in benchmark: 2
    Time: 0.5136146903038025min
    Cost: $0.0332325
    --------------------------------
    Bot: Benchmark for CustomBot
    Score: 0.7516094970590276
    Num reports in benchmark: 2
    Time: 0.4508770227432251min
    Cost: $0.03279


The ideal number of questions to get a good sense of whether one bot is better than another can vary. 100+ should tell your something decent. See [this analysis](https://forum.effectivealtruism.org/posts/DzqSh7akX28JEHf9H/comparing-two-forecasters-in-an-ideal-world) for exploration of the numbers.

As of Dec 20, 2024 the benchmarker automatically selects a random set of questions from Metaculus that:
- Are binary questions (yes/no)
- Are currently open
- Will resolve within 3 months
- Have at least 40 forecasters
- Have a community prediction
- Are not notebook/group/conditional questions

As of last edit there are plans to expand this to numeric and multiple choice, but right now it just benchmarks binary questions.

You can grab these questions without using the Benchmarker by running the below



```python
from forecasting_tools import MetaculusApi

questions = MetaculusApi.get_benchmark_questions(
    num_of_questions_to_return=100,
)
```

You can also save/load benchmarks to/from json


```python
from forecasting_tools import BenchmarkForBot

# Load
file_path = "benchmarks/benchmark.json"
benchmarks: list[BenchmarkForBot] = BenchmarkForBot.load_json_from_file_path(file_path)

# Save
new_benchmarks: list[BenchmarkForBot] = benchmarks
BenchmarkForBot.save_object_list_to_file_path(new_benchmarks, file_path)

# To/From Json String
single_benchmark = benchmarks[0]
json_object: dict = single_benchmark.to_json()
new_benchmark: BenchmarkForBot = BenchmarkForBot.from_json(json_object)

# Note: Make sure to set the 'FILE_WRITING_ALLOWED' environment variable to true if you want to save the benchmarks to a file

```

## Smart Searcher
The Smart Searcher acts like an LLM with internet access. It works a lot like Perplexity.ai API, except:
- It has clickable citations that highlights and links directly to the paragraph cited using text fragments
- You can ask the AI to use filters for domain, date, and keywords
- There are options for structured output (Pydantic objects, lists, dict, list\[dict\], etc.)
- Concurrent search execution for faster results
- Optional detailed works cited list


```python

searcher = SmartSearcher(
    temperature=0,
    num_searches_to_run=2,
    num_sites_per_search=10,  # Results returned per search
    include_works_cited_list=False  # Add detailed citations at the end
)

response = await searcher.invoke(
    "What is the recent news for Apple?"
)

print(response)
```

Example output:
> Recent news about Apple includes several significant developments:
>
> 1. **Expansion in India**: Apple is planning to open four more stores in India, with two in Delhi and Mumbai, and two in Bengaluru and Pune. This decision follows record revenues in India for the September 2024 quarter, driven by strong iPhone sales. Tim Cook, Apple's CEO, highlighted the enthusiasm and growth in the Indian market during the company's earnings call \[[1](https://telecomtalk.info/tim-cook-makes-major-announcement-for-apple-in-india/984260/#:~:text=This%20is%20not%20a%20new,first%20time%20Apple%20confirmed%20it.)\]\[[4](https://telecomtalk.info/tim-cook-makes-major-announcement-for-apple-in-india/984260/#:~:text=This%20is%20not%20a%20new,set%20an%20all%2Dtime%20revenue%20record.)\]\[[5](https://telecomtalk.info/tim-cook-makes-major-announcement-for-apple-in-india/984260/#:~:text=Previously%2C%20Diedre%20O%27Brien%2C%20Apple%27s%20senior,East%2C%20India%20and%20South%20Asia.)\]\[[8](https://telecomtalk.info/tim-cook-makes-major-announcement-for-apple-in-india/984260/#:~:text=At%20the%20company%27s%20earnings%20call,four%20new%20stores%20in%20India.)\].
>
> 2. **Product Launches**: Apple is set to launch new iMac, Mac mini, and MacBook Pro models with M4 series chips on November 8, 2024. Additionally, the Vision Pro headset will be available in South Korea and the United Arab Emirates starting November 15, 2024. The second season of the Apple TV+ sci-fi series "Silo" will also premiere on November 15, 2024 \[[2](https://www.macrumors.com/2024/11/01/what-to-expect-from-apple-this-november/#:~:text=And%20the%20Vision%20Pro%20launches,the%20App%20Store%2C%20and%20more.)\]\[[12](https://www.macrumors.com/2024/11/01/what-to-expect-from-apple-this-november/#:~:text=As%20for%20hardware%2C%20the%20new,announcements%20in%20store%20this%20November.)\].
>
> ... etc ...

You can also use structured outputs by providing a Pydantic model (or any other simpler type hint) and using the schema formatting helper:


```python
from pydantic import BaseModel, Field
from forecasting_tools import SmartSearcher

class Company(BaseModel):
    name: str = Field(description="Full company name")
    market_cap: float = Field(description="Market capitalization in billions USD")
    key_products: list[str] = Field(description="Main products or services")
    relevance: str = Field(description="Why this company is relevant to the search")

searcher = SmartSearcher(temperature=0, num_searches_to_run=4, num_sites_per_search=10)

schema_instructions = searcher.get_schema_format_instructions_for_pydantic_type(Company)
prompt = f"""Find companies that are leading the development of autonomous vehicles.
Return as a list of companies with their details. Remember to give me a list of the schema provided.

{schema_instructions}"""

companies = await searcher.invoke_and_return_verified_type(prompt, list[Company])

for company in companies:
    print(f"\n{company.name} (${company.market_cap}B)")
    print(f"Relevance: {company.relevance}")
    print("Key Products:")
    for product in company.key_products:
        print(f"- {product}")
```

The schema instructions will format the Pydantic model into clear instructions for the AI about the expected output format and field descriptions.


## Key Factors Researcher
The Key Factors Researcher helps identify and analyze key factors that should be considered for a forecasting question. As of last update, this is the most reliable of the tools, and gives something useful and accurate almost every time. It asks a lot of questions, turns search results into a long list of bullet points, rates each bullet point on ~8 criteria, and returns the top results.


```python
from forecasting_tools import KeyFactorsResearcher, BinaryQuestion, QuestionState, ScoredKeyFactor

# Consider using MetaculusApi.get_question_by_id or MetaculusApi.get_question_by_url instead
question = BinaryQuestion(
    question_text="Will YouTube be blocked in Russia?",
    background_info="...", # Or 'None'
    resolution_criteria="...", # Or 'None'
    fine_print="...", # Or 'None'
    id_of_post=0, # The ID and state only matters if using Metaculus API calls
    state=QuestionState.OPEN
)

# Find key factors
key_factors = await KeyFactorsResearcher.find_and_sort_key_factors(
    metaculus_question=question,
    num_key_factors_to_return=5,  # Number of final factors to return
    num_questions_to_research_with=26  # Number of research questions to generate
)

print(ScoredKeyFactor.turn_key_factors_into_markdown_list(key_factors))
```

Example output:
> - The Russian authorities have slowed YouTube speeds to near unusable levels, indicating a potential groundwork for a future ban. [Source Published on 2024-09-12](https://meduza.io/en/feature/2024/09/12/the-russian-authorities-slowed-youtube-speeds-to-near-unusable-levels-so-why-are-kremlin-critics-getting-more-views#:~:text=Kolezev%20attributed%20this%20to%20the,suddenly%20stopped%20working%20in%20Russia.)
> - Russian lawmaker Alexander Khinshtein stated that YouTube speeds would be deliberately slowed by up to 70% due to Google's non-compliance with Russian demands, indicating escalating measures against YouTube. [Source Published on 2024-07-25](https://www.yahoo.com/news/russia-slow-youtube-speeds-google-180512830.html#:~:text=Russia%20will%20deliberately%20slow%20YouTube,forces%20and%20promoting%20extremist%20content.)
> - The press secretary of President Vladimir Putin, Dmitry Peskov, denied that the authorities intended to block YouTube, attributing access issues to outdated equipment due to sanctions. [Source Published on 2024-08-17](https://www.wsws.org/en/articles/2024/08/17/pbyj-a17.html#:~:text=%5BAP%20Photo%2FAP%20Photo%5D%20On%20July,two%20years%20due%20to%20sanctions.)
> - YouTube is currently the last Western social media platform still operational in Russia, with over 93 million users in the country. [Source Published on 2024-07-26](https://www.techradar.com/pro/vpn/youtube-is-getting-throttled-in-russia-heres-how-to-unblock-it#:~:text=If%20you%27re%20in%20Russia%20and,platform%20to%20work%20in%20Russia.)
> - Russian users reported mass YouTube outages amid growing official criticism, with reports of thousands of glitches in August 2024. [Source Published on 2024-08-09](https://www.aljazeera.com/news/2024/8/9/russian-users-report-mass-youtube-outage-amid-growing-official-criticism?traffic_source=rss#:~:text=Responding%20to%20this%2C%20a%20YouTube,reported%20about%20YouTube%20in%20Russia.)


The simplified pydantic structure of the scored key factors is:
```python
class ScoredKeyFactor():
    text: str
    factor_type: KeyFactorType (Pro, Con, or Base_Rate)
    citation: str
    source_publish_date: datetime | None
    url: str
    score_card: ScoreCard
    score: int
    display_text: str
```

## Base Rate Researcher
The Base Rate Researcher helps calculate historical base rates for events. As of last update, it gives decent results around 50% of the time. It orchestrates the Niche List Researcher and the Fermi Estimator to find base rate.


```python
from forecasting_tools import BaseRateResearcher

# Initialize researcher
researcher = BaseRateResearcher(
    "How often has Apple been successfully sued for patent violations?"
)

# Get base rate analysis
report = await researcher.make_base_rate_report()

print(f"Historical rate: {report.historical_rate:.2%}")
print(report.markdown_report)
```

## Niche List Researcher
The Niche List Researcher helps analyze specific lists of events or items. The researcher will:
1. Generate a comprehensive list of potential matches
2. Remove duplicates
3. Fact check each item against multiple criteria
4. Return only validated items (unless include_incorrect_items=True)


```python
from forecasting_tools import NicheListResearcher

researcher = NicheListResearcher(
    type_of_thing_to_generate="Times Apple was successfully sued for patent violations between 2000-2024"
)

fact_checked_items = await researcher.research_niche_reference_class(
    return_invalid_items=False
)

for item in fact_checked_items:
    print(item)
```

The simplified pydantic structure of the fact checked items is:
```python
class FactCheckedItem():
    item_name: str
    description: str
    is_uncertain: bool | None = None
    initial_citations: list[str] | None = None
    fact_check: FactCheck
    type_description: str
    is_valid: bool
    supporting_urls: list[str]
    one_line_fact_check_summary: str

class FactCheck(BaseModel):
    criteria_assessments: list[CriteriaAssessment]
    is_valid: bool

class CriteriaAssessment():
    short_name: str
    description: str
    validity_assessment: str
    is_valid_or_unknown: bool | None
    citation_proving_assessment: str | None
    url_proving_assessment: str | None:
```

## Fermi Estimator
The Fermi Estimator helps break down numerical estimates using Fermi estimation techniques.



```python
from forecasting_tools import Estimator

estimator = Estimator(
    type_of_thing_to_estimate="books published worldwide each year",
    previous_research=None  # Optional: Pass in existing research
)

size, explanation = await estimator.estimate_size()

print(f"Estimate: {size:,}")
print(explanation)
```

Example output (Fake data with links not added):
> I estimate that there are 2,750,000 'books published worldwide each year'.
>
> **Facts**:
> - Traditional publishers release approximately 500,000 new titles annually in English-speaking countries [1]
> - China publishes around 450,000 new books annually [2]
> - The global book market was valued at $92.68 billion in 2023 [3]
> - Self-published titles have grown by 264% in the last 5 years [4]
> - Non-English language markets account for about 50% of global publishing [5]
>
> **Estimation Steps and Assumptions**:
> 1. Start with traditional English publishing: 500,000 titles
> 2. Add Chinese market: 500,000 + 450,000 = 950,000
> 3. Account for other major languages (50% of market): 950,000 * 2 = 1,900,000
> 4. Add self-published titles (estimated 45% of total): 1,900,000 * 1.45 = 2,755,000
>
> **Background Research**: [Additional research details...]

## Metaculus API
The Metaculus API wrapper helps interact with Metaculus questions and tournaments. Grabbing questions returns a pydantic object, and supports important information for Binary, Multiple Choice, Numeric,and Date questions.


```python
from forecasting_tools import MetaculusApi, ApiFilter
from datetime import datetime

question = MetaculusApi.get_question_by_post_id(578)
print(question.page_url)

question = MetaculusApi.get_question_by_url("https://www.metaculus.com/questions/578/human-extinction-by-2100/")
print(question.page_url)

questions = MetaculusApi.get_all_open_questions_from_tournament(
    tournament_id=MetaculusApi.CURRENT_QUARTERLY_CUP_ID
)
print(f"Num tournament questions: {len(questions)}")

api_filter = ApiFilter(
    num_forecasters_gte=40,
    close_time_gt=datetime(2023, 12, 31),
    close_time_lt=datetime(2024, 12, 31),
    scheduled_resolve_time_lt=datetime(2024, 12, 31),
    allowed_types=["binary"],
    allowed_statuses=["resolved"],
)
questions = await MetaculusApi.get_questions_matching_filter(100, api_filter, randomly_sample=False)
print(f"Num filtered questions: {len(questions)}")

benchmark_questions = MetaculusApi.get_benchmark_questions(
    num_of_questions_to_return=20
)
print(f"Num benchmark questions: {len(benchmark_questions)}")

MetaculusApi.post_binary_question_prediction(
    question_id=578, # Note that the question ID is not always the same as the post ID
    prediction_in_decimal=0.012  # Must be between 0.01 and 0.99
)
print("Posted prediction")
MetaculusApi.post_question_comment(
    post_id=578,
    comment_text="Here's example reasoning for testing... This will be a private comment..."
)
print("Posted comment")
```

    https://www.metaculus.com/questions/578
    https://www.metaculus.com/questions/578
    Num tournament questions: 0
    Num filtered questions: 100.
    Num benchmark questions: 20.
    Posted prediction
    Posted comment


## Monetary Cost Manager
The Monetary Cost Manager helps to track AI and API costs. It tracks expenses and errors if it goes over the limit. Leave the limit empty to disable the limit. It shouldn't be trusted as an exact expense, but a good estimate of costs. See `forecasting_tools/ai_models/README.md` for more details, and some flaws it has.


```python
from forecasting_tools import MonetaryCostManager
from forecasting_tools import (
    ExaSearcher, Gpt4o, SmartSearcher, Claude35Sonnet, Perplexity
)

max_cost = 5.00

with MonetaryCostManager(max_cost) as cost_manager:
    prompt = "What is the weather in Tokyo?"
    result = await Perplexity().invoke(prompt)
    result = await Gpt4oMetaculusProxy().invoke(prompt)
    result = await Gpt4o().invoke(prompt)
    result = await SmartSearcher().invoke(prompt)
    result = await Claude35Sonnet().invoke(prompt)
    result = await ExaSearcher().invoke(prompt)
    # ... etc ...

    current_cost = cost_manager.current_usage
    print(f"Current cost: ${current_cost:.2f}")
```


# Local Development

## Environment Variables
The environment variables you need can be found in ```.env.template```. Copy this template as ```.env``` and fill it in. As of last update, you only strictly need OPENAI_API_KEY and EXA_API_KEY.

## Docker Dev Container
Dev containers are reliable ways to make sure environments work on everyone's machine the first try and so you don't have to spend hours setting up your environment (especially if you have Docker already installed). If you would rather just use poetry, without the dev container, you can skip to "Alternatives to Docker". Otherwise, to get your development environment up and running, you need to have Docker Engine installed and running. Once you do, you can use the VSCode dev container pop-up to automatically set up everything for you.

### Install Docker
For Windows and Mac, you will download Docker Desktop. For Linux, you will download Docker Engine. (NOTE: These instructions might be outdated).

First download and setup Docker Engine using the instructions at the link below for your OS:
 * Windows: [windows-install](https://docs.docker.com/desktop/install/windows-install/)
 * Mac: [mac-install](https://docs.docker.com/desktop/install/mac-install/)
 * Linux: [install](https://docs.docker.com/engine/install/)
    * Do not install Docker Desktop for Linux, rather, select your Linux distribution on the left sidebar and follow the distribution specific instructions for Docker engine. Docker Desktop runs with a different environment in Linux.
    * Remember to follow the post-installation steps for Linux: [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/)


### Starting the container
Once Docker is installed, when you open up the project folder in VSCode, you will see a pop up noting that you have a setup for a dev container, and asking if you would like to open the folder in a container. You will want to click "open in container". This will automatically set up everything you need and bring you into the container. If the Docker process times out in the middle of installing python packages you can run the `.devcontiner/postinstall.sh` manually. You may need to have the VSCode Docker extension and/or devcontainer extension downloaded in order for the pop up to appear.

Once you are in the container, poetry should have already installed a virtual environment. For VSCode features to use this environment, you will need to select the correct python interpreter. You can do this by pressing `Ctrl + Shift + P` and then typing `Python: Select Interpreter`. Then select the interpreter that starts with `.venv`.

A number of vscode extensions are installed automatically (e.g. linting). You may need to wait a little while and then reload the window after all of these extensions are installed. You can install personal vscode extensions in the dev environment.


### Managing Docker
There are many ways to manager Docker containers, but generally if you download the vscode Docker extension, you will be able to stop/start/remove all containers and images.


### Alternatives to Docker
If you choose not to run Docker, you can use poetry to set up a local virtual environment. If you are on Ubuntu, you should be able to just read through and then run `.devcontainer/postinstall.sh`. If you aren't on Ubuntu, check out the links in the postinstall file for where install instructions for dependencies were originally found. You may also want to take a look at VSCode extensions that would be installed (see the list in the `.devcontainer/devcontainer.json` file) so that some VSCode workplace settings work out of the box (e.g. automatic Black Formatting).

## Running the Front End
You can run any front end folder in the front_end directory by executing `streamlit run front_end/Home.py`. This will start a development server for you that you can run.

Streamlit makes it very easy to publish demos. If you change up the bot(s) within the front_end, you can quickly deploy a free demo others can play with (look up streamlit cloud).

## Testing
This repository uses pytest tests are subdivided into folders 'unit_tests', 'low_cost_or_live_api', 'expensive'. Unit tests should always pass, while the other tests are for sanity checking. The low cost folder should be able to be run on mass without a huge cost to you. Do not run `pytest` without specifying which folder you want or else you will incur some large expenses from the 'expensive' folder.

Also it's helpful to use the log file that is automatically placed at `logs/latest.log`. You may need to set `FILE_WRITING_ALLOWED=TRUE` in .env for this to work.

# Contributing

## Getting Started

1. **Fork the Repository**: Fork the repository on GitHub. Clone your fork locally: `git clone git@github.com:your-username/forecasting-tools.git`
2. **Set Up Development Environment**: Follow the "Local Development" section in the README to set up your environment
3. **Come up with an improvement**: Decide on something worth changing. Perhaps, you want to add your own custom bot to the forecasting_bots folder. Perhaps you want to add a tool that you think others could benefit from. As long as your code is clean, you can probably expect your contribution to be accepted, though if you are worried about adoption, feel free to chat on our discord or create an issue.
4. **Make a pull request**:
   - Make changes
   - Push your changes to your fork
   - Make sure you rebase with the upstream main branch before doing a PR (`git fetch upstream` and `git rebase upstream/main`)
   - Go to your fork in github, and choose the branch that you have that has your changes
   - You should see a 'Contribute' button. Click this and make a pull request.
   - Fill out the pull request template with a  description of what changed and why and Url for related issues
   - Request review from maintainers
   - Respond to any feedback and make requested changes

## Development Guidelines

1. **Code Style**
   - Code is automatically formatted on commit using precommit (this should be automatically installed when you start the devcontainer). Formatting is done using Black, Ruff, and some other tools.
   - Use type hints for all function parameters and return values
   - Use descriptive variable names over comments
   - Follow existing patterns in the codebase

2. **Testing**
   - Add tests where appropriate for new functionality. We aren't shooting for full code coverage, but you shouldn't make none.
   - Run unit tests locally before merging to check if you broke anything. See the 'Testing' section.

## Questions or Issues?

- Join our [Discord](https://discord.gg/Dtq4JNdXnw) for questions
- Open an issue for bugs or feature requests

Thank you for helping improve forecasting-tools!
