import streamlit as st
import pandas as pd
import altair as alt
import os

# Page configuration
st.set_page_config(
    page_title="Aviation Sustainability Dashboard - Main Categories",
    layout="wide"
)

# Title
st.title("Aviation Sustainability Main Categories")

# Add debug info
st.sidebar.subheader("Debug Information")
st.sidebar.write(f"Current working directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir()}")

# Create fallback data in case CSV is missing
fallback_data = {
    "Category": ["Innovating", "Operationalizing", "Prognosticating", "Synchronizing"],
    "Total_Term_Frequency": [51373, 8833, 7778, 3994],
    "N_Cases": [705, 595, 525, 436],
    "TFIDF": [5.665513023284259, 3.6725261028497447, 5.519391664177306, 5.114374824487207]
}

# Try to load CSV data, use fallback if not found
try:
    # First try the expected location
    df = pd.read_csv('main_category_summary.csv')
    st.sidebar.success("Successfully loaded CSV file!")
except FileNotFoundError:
    st.sidebar.warning("Could not find main_category_summary.csv - using fallback data")
    # If that fails, try data directory
    try:
        df = pd.read_csv('./data/main_category_summary.csv')
        st.sidebar.success("Successfully loaded CSV from data directory!")
    except FileNotFoundError:
        # If all fails, use the fallback data
        df = pd.DataFrame(fallback_data)
        st.sidebar.warning("Using hardcoded fallback data as CSV could not be found")

# Display raw data in sidebar for debugging
st.sidebar.subheader("Raw Data")
st.sidebar.dataframe(df)

# Metric selector
metric_type = st.radio(
    "Select Metric:",
    ["Total_Term_Frequency", "N_Cases", "TFIDF"],
    horizontal=True,
    format_func=lambda x: {
        "Total_Term_Frequency": "Term Frequency",
        "N_Cases": "Number of Cases",
        "TFIDF": "TF-IDF Score"
    }[x]
)

# Sort data based on selected metric
df_sorted = df.sort_values(by=metric_type, ascending=False).reset_index(drop=True)

# Create two columns for the charts
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Category Rankings")
    
    # Bar chart with Altair
    chart_title = {
        "Total_Term_Frequency": "Total Term Frequency by Category",
        "N_Cases": "Number of Cases by Category",
        "TFIDF": "TF-IDF Score by Category"
    }[metric_type]
    
    chart = alt.Chart(df_sorted).mark_bar().encode(
        y=alt.Y('Category:N', sort=None, title=None),
        x=alt.X(f'{metric_type}:Q', title={
            "Total_Term_Frequency": "Total Term Frequency",
            "N_Cases": "Number of Cases",
            "TFIDF": "TF-IDF Score"
        }[metric_type]),
        color=alt.Color('Category:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Category', metric_type]
    ).properties(
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Distribution")
    
    # Create pie chart with Altair
    # First need to calculate angles and positions for pie segments
    pie_data = df_sorted.copy()
    pie_data['angle'] = pie_data[metric_type] / pie_data[metric_type].sum() * 2 * 3.14159
    pie_data['percentage'] = (pie_data[metric_type] / pie_data[metric_type].sum() * 100).round(1).astype(str) + '%'
    
    # Add a column for plotting
    pie_data = pie_data.reset_index(drop=True)
    
    # Create the chart
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta='angle:Q',
        color=alt.Color('Category:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Category', metric_type, 'percentage']
    ).properties(
        width=300,
        height=300
    )
    
    st.altair_chart(pie_chart, use_container_width=True)

# Key insights section
st.subheader("Key Insights:")
st.markdown("""
- **Innovating** has the highest term frequency and appears in the most cases, indicating it's the dominant topic.
- **Operationalizing** has the lowest TF-IDF score, suggesting less distinctive terminology.
- **Prognosticating** has a relatively high TF-IDF score despite lower frequency, showing specialized terminology.
- **Synchronizing** appears in the fewest documents but maintains a competitive TF-IDF score.
""")
