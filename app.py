import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Aviation Sustainability Dashboard",
    layout="wide"
)

# Title
st.title("Aviation Sustainability Categories")

# Create DataFrame from data
data = {
    "Category": [
        "Sustainable Aviation Fuels",
        "Electrification of Aviation",
        "Advanced Air Mobility",
        "Hydrogen Powered Aviation",
        "Aircraft Efficiency",
        "Innovating",
        "Prognosticating",
        "Operationalizing",
        "Synchronizing"
    ],
    "Total_Term_Frequency": [
        67861,
        27678,
        14624,
        13478,
        9232,
        8473,
        7010,
        5300,
        3750
    ],
    "N_Cases": [
        606,
        675,
        558,
        562,
        499,
        561,
        513,
        537,
        426
    ],
    "TFIDF": [
        27.55,
        5.66,
        8.61,
        7.71,
        8.14,
        4.88,
        5.64,
        3.62,
        5.27
    ]
}

df = pd.DataFrame(data)

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
        color=alt.Color('Category:N', legend=None),
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
- **Sustainable Aviation Fuels** has the highest term frequency and TF-IDF score, suggesting it's the dominant topic in aviation sustainability discussions.
- **Electrification of Aviation** appears in the most cases, indicating widespread but potentially less focused discussion.
- Despite lower term frequency, **Advanced Air Mobility** and **Aircraft Efficiency** have high TF-IDF scores, suggesting they're distinctive topics within their documents.
- Process-oriented categories like **Operationalizing** and **Synchronizing** show lower metrics overall, indicating they're less prominent in the discourse.
""")

# Add requirements.txt guidance
st.sidebar.title("Deployment Info")
st.sidebar.info("""
To deploy this dashboard to Streamlit Community Cloud:

1. Save this code as `app.py`
2. Create a `requirements.txt` file with:
   ```
   streamlit
   pandas
   altair
   ```
3. Push both files to a GitHub repository
4. Go to share.streamlit.io and connect to your repository
""")