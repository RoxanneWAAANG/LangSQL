# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

"""
Module: streamlit_app
Implements a Streamlit front-end for a Text-to-SQL chatbot, supporting question logging and dynamic SQL generation via ChatBot.
"""
import os
import streamlit as st

from text2sql import ChatBot
from utils.db_utils import add_a_record


class Text2SQLApp:
    """
    Encapsulates the Streamlit interface for interacting with a Text-to-SQL ChatBot.
    """
    def __init__(self) -> None:
        """
        Initialize the Text2SQLChatBot.
        """
        self.bot = ChatBot()
        self.db_schemas = self._load_demo_schemas()

    @staticmethod
    def _load_demo_schemas() -> dict:
        """
        Provide hard-coded demo database schemas for the sidebar display.

        Returns:
            A mapping from database ID to DDL string.
        """
        return {
            "singer": """
    CREATE TABLE \"singer\" (
        \"Singer_ID\" int,
        \"Name\" text,
        \"Birth_Year\" real,
        \"Net_Worth_Millions\" real,
        \"Citizenship\" text,
        PRIMARY KEY (\"Singer_ID\")
    );

    CREATE TABLE \"song\" (
        \"Song_ID\" int,
        \"Title\" text,
        \"Singer_ID\" int,
        \"Sales\" real,
        \"Highest_Position\" real,
        PRIMARY KEY (\"Song_ID\"),
        FOREIGN KEY (\"Singer_ID\") REFERENCES \"singer\"(\"Singer_ID\")
    );
    """,
            # Add more schemas here as needed
        }

    def run(self) -> None:
        """
        Build and run the Streamlit UI.
        """
        st.title("Text-to-SQL Chatbot")
        self._render_sidebar()

        question = st.text_input("Enter your question:")
        selected_db = st.session_state.get("selected_db")

        if question and selected_db:
            self._log_question(question, selected_db)
            response = self.bot.get_response(question, selected_db)
            st.write(f"**Database:** {selected_db}")
            st.write(f"**Results:** {response}")

    def _render_sidebar(self) -> None:
        """
        Render the sidebar elements: database selector and schema display.
        """
        st.sidebar.header("Select a Database")
        db_ids = list(self.db_schemas.keys())
        st.sidebar.selectbox(
            "Choose a database:",
            db_ids,
            key="selected_db",
            index=0
        )
        selected = st.session_state.get("selected_db")
        schema = self.db_schemas.get(selected, "")
        st.sidebar.text_area(
            "Database Schema", schema, height=400
        )

    def _log_question(self, question: str, db_id: str) -> None:
        """
        Persist the user question for auditing or analytics.

        Args:
            question: User's natural language input.
            db_id: Target database identifier.
        """
        try:
            add_a_record(question, db_id)
        except Exception as e:
            st.warning(f"Failed to log question: {e}")


def main():
    """
    Entry point for Streamlit. Launches the app.
    """
    app = Text2SQLApp()
    app.run()


if __name__ == "__main__":
    main()