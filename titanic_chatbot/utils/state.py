import streamlit as st
from utils.prompts import system_prompt_template
from utils.questions import all_suggested_questions
import random

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = system_prompt_template

    if 'code_cache' not in st.session_state:
        st.session_state.code_cache = {}

    if 'explanation_cache' not in st.session_state:
        st.session_state.explanation_cache = {}

    if 'followups_cache' not in st.session_state:
        st.session_state.followups_cache = {}

    if "sampled_questions" not in st.session_state:
        st.session_state.sampled_questions = random.sample(all_suggested_questions, 4)
