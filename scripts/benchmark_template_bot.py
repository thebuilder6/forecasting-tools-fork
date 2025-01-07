from forecasting_tools import Benchmarker, BenchmarkForBot, TemplateBot


async def benchmark_template_bot() -> None:
    # Run benchmark on multiple bots
    bots = [TemplateBot()]  # Add your custom bots here
    benchmarker = Benchmarker(
        forecast_bots=bots,
        number_of_questions_to_use=2,  # Recommended 100+ for meaningful results
        file_path_to_save_reports="benchmarks/",
    )
    benchmarks: list[BenchmarkForBot] = await benchmarker.run_benchmark()

    # View results
    for benchmark in benchmarks:
        print(f"Bot: {benchmark.name}")
        print(
            f"Score: {benchmark.average_inverse_expected_log_score}"
        )  # Lower is better
        print(f"Num Forecasts: {len(benchmark.forecast_reports)}")
        print(f"Time: {benchmark.time_taken_in_minutes}min")
        print(f"Cost: ${benchmark.total_cost}")


if __name__ == "__main__":
    asyncio.run(benchmark_template_bot())
