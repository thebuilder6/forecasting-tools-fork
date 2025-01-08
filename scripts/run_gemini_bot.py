import asyncio

from gemini_bot import GeminiBot

from forecasting_tools import MetaculusApi


async def main():
    bot = GeminiBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="logs/forecasts/",
    )

    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
    )

    reports = await bot.forecast_questions([question])

    for report in reports:
        print(f"\nQuestion: {report.question.question_text}")
        print(f"Prediction: {report.prediction}")
        print("\nReasoning:")
        print(report.explanation)


if __name__ == "__main__":
    asyncio.run(main())
