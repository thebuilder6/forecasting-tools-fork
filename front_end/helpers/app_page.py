from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import streamlit as st
from streamlit.navigation.page import StreamlitPage

logger = logging.getLogger(__name__)


class AppPage(ABC):
    PAGE_DISPLAY_NAME: str = NotImplemented
    URL_PATH: str = NotImplemented
    IS_DEFAULT_PAGE: bool = False

    def __init_subclass__(cls: type[AppPage], *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        is_abstract = ABC in cls.__bases__
        if not is_abstract:
            if cls.PAGE_DISPLAY_NAME is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define PAGE_DISPLAY_NAME"
                )
            if cls.URL_PATH is NotImplemented:
                raise NotImplementedError("You forgot to define URL_PATH")

    @classmethod
    def main(cls) -> None:
        cls.header()
        asyncio.run(cls._async_main())
        cls.footer()

    @classmethod
    @abstractmethod
    async def _async_main(cls) -> None:
        pass

    @classmethod
    def convert_to_streamlit_page(cls) -> StreamlitPage:
        page = st.Page(
            cls.main,
            title=cls.PAGE_DISPLAY_NAME,
            icon=None,
            url_path=cls.URL_PATH,
            default=cls.IS_DEFAULT_PAGE,
        )
        return page

    @classmethod
    def header(cls):
        pass

    @classmethod
    def footer(cls):
        st.markdown("---")
        st.write(
            "This is the demo site for the "
            "[forecasting-tools python package](https://github.com/CodexVeritas/forecasting-tools)."
        )
        # st.write(
        #     "For those willing to give feedback (even a quick thumbs up or down "
        #     "occasionally), please use the tools as much as you want (I'll give "
        #     "a message if costs become bad). Regular feedback is super valuable. "
        #     "Otherwise please donate to help support the project "
        #     "[☕️ Buy me a coffee](https://buymeacoffee.com/mokoresearch)"
        # )
        st.write(
            "Give feedback on [Discord](https://discord.gg/Dtq4JNdXnw) or email "
            "me at [moko.research@gmail.com](mailto:moko.research@gmail.com). "
            "Let me know what I can do to make this a tool you will want to use "
            "every day! Let me know if you want to chat (I'll Venmo \\$10 to "
            "anyone willing to do a longer 15-30min interview 😉)"
        )
        # st.write(
        #     "Join the [Forecasting Meetup Discord](https://discord.gg/Dtq4JNdXnw) "
        #     "to practice forecasting with real people weekly."
        # )
        st.write(
            "Queries made to the website are saved to a database and may be "
            "reviewed to help improve the tool"
        )
