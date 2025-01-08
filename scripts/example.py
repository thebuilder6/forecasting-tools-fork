import asyncio

from dotenv import load_dotenv

from forecasting_tools import MetaculusApi, TemplateBot


async def main():
    # Initialize the bot
    bot = TemplateBot(
        research_reports_per_question=2,  # Using a smaller number for testing
        predictions_per_research_report=3,
        publish_reports_to_metaculus=False,  # Set to True if you want to post to Metaculus
        folder_to_save_reports_to="logs/forecasts/",
    )

    # Get a specific question from Metaculus
    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
    )

    # Run forecast
    reports = await bot.forecast_questions([question])

    # Print results
    for report in reports:
        print(f"\nQuestion: {report.question.question_text}")
        print(f"Prediction: {report.prediction}")
        print("\nReasoning:")
        print(report.explanation)


# Run the async function
if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
