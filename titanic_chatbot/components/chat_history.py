import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def render_chat_history():
    """Displays the previous Q&A chat history, code, and outputs."""
    if not st.session_state.get("history"):
        return

    # Create two columns: one for title, one for button
    col1, col2 = st.columns([4, 1])

    with col1:
        st.write("#### Titanic Chatbot")

    with col2:
        if st.button("Clear Chat"):
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

        if entry.get("explanation"):
            st.markdown(f"**Explanation:** {entry['explanation']}")

        if entry.get("tokens_used") is not None and entry.get("response_time") is not None:
            st.markdown(
                f"**Tokens Used:** {entry['tokens_used']} &nbsp;&nbsp;&nbsp; **Time Taken:** {entry['response_time']:.2f}s"
            )

        st.markdown("---")
