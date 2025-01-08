from __future__ import annotations

import argparse
import asyncio
import os
import sys

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

import logging

from forecasting_tools import MetaculusApi
from forecasting_bots.my_custom_bot import MyCustomBot

logger = logging.getLogger(__name__)


async def main():
    bot = MyCustomBot(
        research_reports_per_question=3,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="logs/forecasts/",
        skip_previously_forecasted_questions=True
    )
    
    reports = await bot.forecast_on_tournament(
        MetaculusApi.AI_COMPETITION_ID_Q4
    )
    
    for report in reports:
        print(f"Question: {report.question.question_text}")
        print(f"Prediction: {report.prediction}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
