import streamlit as st
import pandas as pd
from utils.llm_handler import generate_code
from utils.executor import execute_code
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Load Titanic dataset
df = pd.read_csv('data/titanic.csv')

# --- Data Cleaning & Imputation ---
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Deck'] = df['Cabin'].str[0]
sex_mapping = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_mapping)
df.drop(columns=['Cabin'], inplace=True)

# --- Initialize session state ---
if 'history' not in st.session_state:
    st.session_state.history = []  # Each entry: {"question": ..., "code": ..., "output": ..., "fig": ...}
    st.session_state.messages = []  # Chat history for LLM

# --- Streamlit UI ---
st.title("🧠 Titanic Data Analyst Bot")
st.markdown("Ask any question about the Titanic dataset.")

st.write("### Cleaned Titanic Dataset Preview")
st.dataframe(df)


# Show full chat history
if st.session_state.history:
    st.write("### 💬 Chat History")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()
    for i, entry in enumerate(st.session_state.history):
        st.markdown(f"**Q{i+1}: {entry['question']}**")
        st.code(entry['code'], language='python')
        if entry['output']:
            st.text(entry['output'])
        if entry['fig'] is not None:
            if isinstance(entry['fig'], plt.Figure):
                st.pyplot(entry['fig'])
            elif isinstance(entry['fig'], (go.Figure, px.Figure)):
                st.plotly_chart(entry['fig'], use_container_width=True)
        st.markdown("---")

# Clear input box before rendering widget
if st.session_state.get("clear_input_box"):
    st.session_state["user_question"] = ""
    st.session_state["clear_input_box"] = False

user_input = st.text_input("Ask your next question:", key="user_question")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        code, assistant_msg = generate_code(user_input, df.columns, st.session_state.messages)
        output, fig = execute_code(code, df)
        st.session_state.messages.append(assistant_msg)

        st.session_state.history.append({
            "question": user_input,
            "code": code,
            "output": output,
            "fig": fig
        })

        st.session_state["clear_input_box"] = True  # ✅ Clear input after submit
        st.rerun()
