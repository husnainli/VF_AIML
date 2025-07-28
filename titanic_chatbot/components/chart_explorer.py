# components/chart_explorer.py

import streamlit as st
import plotly.express as px

@st.cache_data(show_spinner=False)
def get_chart_metadata(df):
    excluded_columns = ["PassengerId", "Surname", "FullName", "Ticket", "Name"]
    usable_columns = [col for col in df.columns if col not in excluded_columns]

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

    label_to_column = {v: k for k, v in column_labels.items()}
    column_display_names = [column_labels.get(col, col) for col in usable_columns]

    numeric_columns = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in excluded_columns]
    numeric_display_names = [column_labels.get(col, col) for col in numeric_columns]

    value_mappings = {
        "Survived": {0: "No", 1: "Yes"},
        "Pclass": {1: "1st Class", 2: "2nd Class", 3: "3rd Class"},
        "Sex": {0: "Male", 1: "Female"},
        "IsMarried": {0: "No", 1: "Yes"},
    }

    return column_labels, label_to_column, column_display_names, numeric_display_names, value_mappings


def render_chart_explorer(df):
    column_labels, label_to_column, column_display_names, numeric_display_names, value_mappings = get_chart_metadata(df)

    with st.expander("🕹️ Interactive Chart Explorer", expanded=False):
        st.markdown("Use the dropdowns below to visualize relationships between Titanic dataset variables.")

        y_display_options = ["Number of Passengers"] + numeric_display_names

        x_display = st.selectbox("Select X-axis", options=column_display_names, index=column_display_names.index("Sex") if "Sex" in column_display_names else 0)
        y_display = st.selectbox("Select Y-axis", options=y_display_options, index=0)

        x_axis = label_to_column[x_display]
        y_axis = label_to_column.get(y_display)

        chart_type = st.selectbox(
            "Chart Type",
            options=["Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Violin Plot", "Histogram", "Pie Chart"]
        )

        plot_button = st.button("Generate Chart")

        if plot_button:
            fig = None
            try:
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
