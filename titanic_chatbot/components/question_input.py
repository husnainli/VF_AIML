import streamlit as st
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.llm_handler import generate_code_with_cache, generate_explanation_with_cache
from utils.executor import execute_code
from utils.questions import all_suggested_questions

def render_question_input_and_processing(df):
    # Clear input box before rendering widget
    if st.session_state.get("clear_input_box"):
        st.session_state["user_question"] = ""
        st.session_state["clear_input_box"] = False

    st.write("#### Ask your next question:")
    user_input = st.text_input("Enter a data-related question (tick the box below to include explanation)", key="user_question")

    # Automatically submit if a suggested question was clicked
    if st.session_state.get("trigger_submit"):
        user_input = st.session_state["user_question"]
        st.session_state["trigger_submit"] = False  # Reset trigger
        submit_clicked = True
    else:
        submit_clicked = st.button("Submit")

    generate_explanation_checkbox = st.checkbox("Generate Analysis", value=True)

    if submit_clicked and user_input:
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": user_input})

            code, assistant_msg, code_usage, code_duration, code_cached = generate_code_with_cache(
                user_input,
                st.session_state.messages,
                st.session_state.system_prompt,
                st.session_state.code_cache,
            )

            output, fig, plot_metadata = execute_code(code, df)

            # Conditional explanation generation
            explanation = None
            explanation_usage = type('', (), {'total_tokens': 0})()
            explanation_duration = 0

            if generate_explanation_checkbox:
                explanation, explanation_usage, explanation_duration, explanation_cached = generate_explanation_with_cache(
                    user_input, code, output, plot_metadata, st.session_state.explanation_cache
                )

            st.session_state.messages.append(assistant_msg)

            # Show token/time only if at least one component wasn't cached
            if (not code_cached) or (generate_explanation_checkbox and not explanation_cached):
                total_tokens = code_usage.total_tokens + explanation_usage.total_tokens
                total_time = code_duration + explanation_duration
            else:
                total_tokens = None
                total_time = None

            st.session_state.history.append({
                "question": user_input,
                "code": code,
                "output": output,
                "fig": fig,
                "explanation": explanation if generate_explanation_checkbox else None,
                "tokens_used": total_tokens if total_tokens is not None else None,
                "response_time": total_time if total_time is not None else None
            })

            # Refresh suggested questions after each answer
            st.session_state.sampled_questions = random.sample(all_suggested_questions, 4)

            st.session_state["clear_input_box"] = True
            st.rerun()
