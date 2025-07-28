import streamlit as st
import random
from utils.questions import all_suggested_questions

def render_suggested_questions_right(df):
    main_col, spacer, right_col = st.columns([4, 0.2, 1])

    with right_col:
        st.markdown("---")
        st.markdown("### Suggested Questions")

        for q in st.session_state.sampled_questions:
            if st.button(q, key=f"suggested_right_{q}"):
                st.session_state["user_question"] = q
                st.session_state["trigger_submit"] = True
                st.rerun()

    with spacer:
        st.markdown("""
            <div style='height: 100%; border-left: 1px solid #ccc; margin: 0 auto;'></div>
        """, unsafe_allow_html=True)

    return main_col
