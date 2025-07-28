import streamlit as st
import pandas as pd
import random
from utils.llm_handler import generate_code_with_cache
from utils.llm_handler import generate_explanation_with_cache
from utils.executor import execute_code
from utils.prompts import system_prompt_template
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.data import load_and_prepare_data

# Custom CSS for layout enhancement
st.markdown("""
    <style>
        /* Remove padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0;
        }
        /* Align left + main + right columns with vertical separation */
        .custom-container {
            display: flex;
            gap: 1.5rem;
        }
        .sidebar-left, .main-area, .sidebar-right {
            padding: 1rem;
            background-color: #fafafa;
            border-radius: 10px;
        }
        .sidebar-left {
            flex: 1;
            border-right: 2px solid #eee;
        }
        .main-area {
            flex: 2.5;
        }
        .sidebar-right {
            flex: 1;
            border-left: 2px solid #eee;
        }
        .suggested-btn {
            width: 100%;
            margin: 0.25rem 0;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

df = load_and_prepare_data()

# Columns to exclude from UI
excluded_columns = ["PassengerId", "Surname", "FullName", "Ticket", "Name"]
usable_columns = [col for col in df.columns if col not in excluded_columns]

# Display-friendly column names
column_labels = {
    "Pclass": "Passenger Class",
    "Sex": "Sex",
    "Age": "Age",
    "SibSp": "Siblings/Spouses Aboard",
    "Parch": "Parents/Children Aboard",
    "Fare": "Fare",
    "Embarked": "Port of Embarkation",
    "Cabin": "Cabin",
    "Survived": "Survived",
    "Title": "Title",
    "Deck": "Deck",
    "IsMarried": "Marital Status (Yes/No)"
}

# Reverse mapping for dropdown selection
label_to_column = {v: k for k, v in column_labels.items()}
column_display_names = [column_labels.get(col, col) for col in usable_columns]

numeric_columns = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in excluded_columns]
numeric_display_names = [column_labels.get(col, col) for col in numeric_columns]

# Categorical value mappings (for visualization only)
value_mappings = {
    "Survived": {0: "No", 1: "Yes"},
    "Pclass": {1: "1st Class", 2: "2nd Class", 3: "3rd Class"},
    "Sex": {0: "Male", 1: "Female"},
    "IsMarried": {0: "No", 1: "Yes"},
}

# --- Initialize session state ---
if 'history' not in st.session_state:
    st.session_state.history = []  # Each entry: {"question": ..., "code": ..., "output": ..., "fig": ...}
    st.session_state.messages = []  # Chat history for LLM

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = system_prompt_template

if 'code_cache' not in st.session_state:
    st.session_state.code_cache = {}  # question → (code, assistant_msg, usage, duration)

if 'explanation_cache' not in st.session_state:
    st.session_state.explanation_cache = {}  # (question, code, output, plot_metadata) → (explanation, usage, duration)

if 'followups_cache' not in st.session_state:
    st.session_state.followups_cache = {}  # question → [followup1, followup2, followup3]

# Define the full pool of suggested questions (outside any if)
all_suggested_questions = [
    "What was the survival rate for women?",
    "How many children (age < 12) survived?",
    "Show a pie chart of survival by class.",
    "What is the average age of survivors vs non-survivors?",
    "How many people traveled with family?",
    "What was the average fare by class?",
    "Which age group had the highest survival rate?",
    "Show a histogram of ages of survivors.",
    "What’s the survival rate for each embark point?",
    "How many passengers were traveling alone?",
    "Compare survival of males and females by class.",
    "What is the median age of passengers?",
    "What is the survival rate for passengers under 18?",
    "What percentage of passengers had fare > 50?",
    "Show a bar chart of family size vs survival.",
    "Who were the oldest and youngest survivors?",
    "Did passengers with siblings survive more?",
    "How many passengers had the same last name?",
    "How many survived from 3rd class?",
    "Did females in 1st class survive more than males?",
    "How does survival rate vary by ticket class?",
    "Show survival by age using a violin plot."
]

# Only sample once per session (unless refreshed)
if "sampled_questions" not in st.session_state:
    st.session_state.sampled_questions = random.sample(all_suggested_questions, 4)

with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 0.1rem;'>Quick Insights</h3>", unsafe_allow_html=True)

    # --- Basic Metrics ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Total Passengers**")
        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>{len(df)}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Survival Rate**")
        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>{df['Survived'].mean() * 100:.2f}%</div>", unsafe_allow_html=True)
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Males**")
        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>{(df['Sex'] == 0).sum()}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Females**")
        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>{(df['Sex'] == 1).sum()}</div>", unsafe_allow_html=True)

    # --- Survival Rate by Gender ---
    st.markdown("---")
    st.markdown("**Survival Rate by Gender**")
    gender_map = {0: "Male", 1: "Female"}
    survival_by_gender = df.groupby("Sex")["Survived"].mean()
    col1, col2 = st.columns(2)
    for col, (sex, rate) in zip([col1, col2], survival_by_gender.items()):
        col.markdown(
            f"<div style='text-align: center; font-size: 16px;'>"
            f"<strong>{gender_map[sex]}</strong><br>{rate * 100:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )

    # --- Class Distribution ---
    st.markdown("---")
    st.markdown("**Class Distribution**")
    class_counts = df['Pclass'].value_counts(normalize=True).sort_index()
    col1, col2, col3 = st.columns(3)
    for col, (pclass, pct) in zip([col1, col2, col3], class_counts.items()):
        col.markdown(
            f"<div style='text-align: center; font-size: 16px;'>"
            f"<strong>Class {pclass}</strong><br>{pct * 100:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )

    # --- Survival by Class ---
    st.markdown("**Survival Rate by Class**")
    survival_rates = df.groupby("Pclass")["Survived"].mean().sort_index()
    col1, col2, col3 = st.columns(3)
    for col, (pclass, rate) in zip([col1, col2, col3], survival_rates.items()):
        col.markdown(
            f"<div style='text-align: center; font-size: 16px;'>"
            f"<strong>Class {pclass}</strong><br>{rate * 100:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )

    # --- Age, Family, Fare, Embark ---
    st.markdown("---")
    most_common_age_bin = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 80])
    most_common_range = most_common_age_bin.value_counts().idxmax()
    st.markdown(f"**Common Age Range:** {most_common_range}")
    st.markdown(f"**Avg. Family Size:** {(df['SibSp'] + df['Parch'] + 1).mean():.2f}")
    st.markdown(f"**Top Embark Point:** {df['Embarked'].mode()[0]}")
    st.markdown(f"**Median Fare:** ${df['Fare'].median():.2f}")

# Two-column layout: Main chat + right sidebar
main_col, spacer, right_col = st.columns([4, 0.2, 1])

with right_col:
    st.markdown("### 💡 Suggested Questions")

    # Render the same 4 questions during the session
    for q in st.session_state.sampled_questions:
        if st.button(q, key=f"suggested_right_{q}"):
            st.session_state["user_question"] = q
            st.session_state["trigger_submit"] = True
            st.rerun()

with spacer:
    st.markdown("""
        <div style='height: 100%; border-left: 1px solid #ccc; margin: 0 auto;'></div>
    """, unsafe_allow_html=True)

with main_col:
    st.markdown("""
        <div style='padding-right: 1rem; border-right: 1px solid #ddd;'>
    """, unsafe_allow_html=True)
    # --- Streamlit UI ---
    st.title("🚢 Titanic Data Analyst Dashboard")

    st.write("### Cleaned Titanic Dataset Preview")
    st.dataframe(df)

    with st.expander("🕹️ Interactive Chart Explorer", expanded=False):
        st.markdown("Use the dropdowns below to visualize relationships between Titanic dataset variables.")

        # Inject "Number of Passengers" as virtual Y-axis option
        y_display_options = ["Number of Passengers"] + numeric_display_names

        x_display = st.selectbox("Select X-axis", options=column_display_names, index=column_display_names.index("Sex") if "Sex" in column_display_names else 0)
        y_display = st.selectbox("Select Y-axis", options=y_display_options, index=0)

        x_axis = label_to_column[x_display]
        y_axis = label_to_column.get(y_display)  # Safely get column if not 'Number of Passengers'

        chart_type = st.selectbox(
            "Chart Type",
            options=["Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Violin Plot", "Histogram", "Pie Chart"]
        )

        plot_button = st.button("Generate Chart")

        if plot_button:
            fig = None
            try:
                # Copy and map values for plotting without changing original df
                plot_df = df.copy()
                for col, mapping in value_mappings.items():
                    if col in plot_df.columns:
                        plot_df[col] = plot_df[col].map(mapping).fillna(plot_df[col])

                if y_display == "Number of Passengers":
                    agg_df = plot_df[x_axis].value_counts().reset_index()
                    agg_df.columns = [x_axis, "Passenger Count"]

                    if chart_type == "Bar Chart":
                        fig = px.bar(agg_df, x=x_axis, y="Passenger Count", labels={x_axis: x_display, "Passenger Count": "Number of Passengers"})
                    elif chart_type == "Histogram":
                        fig = px.histogram(plot_df, x=x_axis, labels=column_labels)
                    elif chart_type == "Pie Chart":
                        fig = px.pie(agg_df, names=x_axis, values="Passenger Count", labels=column_labels)
                    else:
                        st.warning("This chart type is not supported when Y-axis is Number of Passengers. Try Bar, Histogram, or Pie.")
                else:
                    # Standard numeric Y-axis charts
                    if chart_type == "Bar Chart":
                        fig = px.bar(plot_df, x=x_axis, y=y_axis, labels=column_labels)
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color="Survived", labels=column_labels)
                    elif chart_type == "Line Chart":
                        fig = px.line(plot_df.sort_values(x_axis), x=x_axis, y=y_axis, labels=column_labels)
                    elif chart_type == "Box Plot":
                        fig = px.box(plot_df, x=x_axis, y=y_axis, color="Survived", labels=column_labels)
                    elif chart_type == "Violin Plot":
                        fig = px.violin(plot_df, x=x_axis, y=y_axis, box=True, color="Survived", labels=column_labels)
                    elif chart_type == "Histogram":
                        fig = px.histogram(plot_df, x=x_axis, color="Survived", labels=column_labels)
                    elif chart_type == "Pie Chart":
                        pie_data = plot_df[x_axis].value_counts().reset_index()
                        pie_data.columns = [x_axis, 'Count']
                        fig = px.pie(pie_data, names=x_axis, values='Count', labels=column_labels)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating chart: {e}")

                    
    # Show full chat history
    if st.session_state.history:
        # Create two columns: one for title, one for button
        col1, col2 = st.columns([4, 1])  # Adjust ratio as needed

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
                st.markdown(f"**Tokens Used:** {entry['tokens_used']} &nbsp;&nbsp;&nbsp; **Time Taken:** {entry['response_time']:.2f}s")
            st.markdown("---")

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