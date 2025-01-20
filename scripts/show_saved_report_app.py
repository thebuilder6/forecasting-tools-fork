import dotenv
import streamlit as st

from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.util.custom_logger import CustomLogger
from front_end.helpers.report_displayer import ReportDisplayer


def main() -> None:
    st.title("Report Viewer")

    # File uploader instead of text area
    file_path = st.text_input("Enter the path to the forecast report file")

    if file_path:
        st.divider()
        st.subheader("Reports:")
        reports = BinaryReport.load_json_from_file_path(file_path)
        ReportDisplayer.display_report_list(reports)


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    main()
