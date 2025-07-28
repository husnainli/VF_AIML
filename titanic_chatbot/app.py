import pandas as pd
import streamlit as st
from utils.state import initialize_session_state
from utils.data import load_and_prepare_data
from utils.css_loader import load_css
from components.sidebar import render_sidebar
from components.chart_explorer import render_chart_explorer
from components.right_panel import render_suggested_questions_right
from components.chat_history import render_chat_history
from components.question_input import render_question_input_and_processing

load_css("styles/layout.css")

df = load_and_prepare_data()

initialize_session_state()

render_sidebar(df)

main_col = render_suggested_questions_right(df)

with main_col:
    st.markdown("""<div style='padding-right: 1rem; border-right: 1px solid #ddd;'>""", unsafe_allow_html=True)
    
    # --- Streamlit UI ---
    st.title("🚢 Titanic Data Analyst Dashboard")

    st.write("### Cleaned Titanic Dataset Preview")
    st.dataframe(df)

    render_chart_explorer(df)
                    
    render_chat_history()

    render_question_input_and_processing(df)

    st.markdown("""<div style="margin-top: 100px;"></div>""",unsafe_allow_html=True)