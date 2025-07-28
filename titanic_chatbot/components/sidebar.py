import streamlit as st
import pandas as pd

def render_sidebar(df):
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
