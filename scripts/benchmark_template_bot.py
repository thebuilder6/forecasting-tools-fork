import asyncio
from forecasting_tools import MetaculusApi, TemplateBot
from forecasting_tools.forecasting.questions_and_reports.forecast_report import ForecastReport

async def run_single_question_benchmark() -> list[ForecastReport]:
    question = MetaculusApi.get_question_by_post_id(1002)
    template_bot = TemplateBot()
    return await template_bot.forecast_question(question)

async def main() -> None:
    print("Forecasting question...")
    forecast_reports = await run_single_question_benchmark()
    print(forecast_reports)

if __name__ == "__main__":
    asyncio.run(main())
