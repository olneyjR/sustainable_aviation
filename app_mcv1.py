import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Aviation Sustainability Dashboard",
    layout="wide"
)

# Title
st.title("Aviation Sustainability Categories")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Main Categories", "Sub Categories", "Main Category Similarity", "Sub Category Similarity"]
)

# Try to load all data files
try:
    # Load main category data
    main_df = pd.read_csv('main_category_summary.csv')
    
    # Load sub-category data
    sub_df = pd.read_csv('sub_category_summary.csv')
    
    # Load main similarity matrix
    main_sim_df = pd.read_csv('main_jaccard_matrix.csv')
    
    # Load sub similarity matrix
    sub_sim_df = pd.read_csv('sub_jaccard_matrix.csv')
    
    # Data loaded successfully
    st.sidebar.success("All data files loaded successfully!")
except FileNotFoundError as e:
    # Show error message but continue with available data
    st.sidebar.error(f"Some data files not found: {str(e)}")
    # Create empty dataframes for missing files
    if 'main_df' not in locals():
        main_df = pd.DataFrame({
            "Category": ["Innovating", "Operationalizing", "Prognosticating", "Synchronizing"],
            "Total_Term_Frequency": [51373, 8833, 7778, 3994],
            "N_Cases": [705, 595, 525, 436],
            "TFIDF": [5.665513023284259, 3.6725261028497447, 5.519391664177306, 5.114374824487207]
        })
    if 'sub_df' not in locals():
        # Create sample sub-category data if file not found
        sub_df = pd.DataFrame({
            "MC": ["Innovating", "Innovating", "Innovating", "Innovating"],
            "PC": ["Aircraft_and_Component_Manufacturing", "Infrastructure_Readiness", 
                   "Safety_and_Reliability", "Sustainable_Aviation_Fuels_and_Propulsion"],
            "Total_Term_Frequency": [4609, 1677, 1257, 43830],
            "N_Cases": [435, 285, 251, 597],
            "TFIDF": [5.94, 5.79, 5.56, 17.92]
        })
    if 'main_sim_df' not in locals():
        main_sim_df = pd.DataFrame()
    if 'sub_sim_df' not in locals():
        sub_sim_df = pd.DataFrame()

# Function for Main Categories page
def show_main_categories():
    st.header("Main Categories Analysis")
    
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
    df_sorted = main_df.sort_values(by=metric_type, ascending=False).reset_index(drop=True)

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

# Function for Sub Categories page
def show_sub_categories():
    st.header("Sub Categories Analysis")
    
    if sub_df.empty:
        st.warning("Sub-category data not available. Please make sure sub_category_summary.csv is in the same directory as this app.")
        return
    
    # Main category selector
    main_categories = sorted(sub_df['MC'].unique())
    selected_main_category = st.selectbox("Select Main Category:", main_categories)
    
    # Filter data for selected main category
    filtered_df = sub_df[sub_df['MC'] == selected_main_category].reset_index(drop=True)
    
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
    df_sorted = filtered_df.sort_values(by=metric_type, ascending=False).reset_index(drop=True)
    
    # Create two columns for the charts
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(f"Sub-Categories for {selected_main_category}")
        
        # Bar chart with Altair
        chart = alt.Chart(df_sorted).mark_bar().encode(
            y=alt.Y('PC:N', sort=None, title=None),
            x=alt.X(f'{metric_type}:Q', title={
                "Total_Term_Frequency": "Total Term Frequency",
                "N_Cases": "Number of Cases",
                "TFIDF": "TF-IDF Score"
            }[metric_type]),
            color=alt.Color('PC:N', scale=alt.Scale(scheme='tableau10')),
            tooltip=['PC', metric_type]
        ).properties(
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Distribution")
        
        # Create pie chart with Altair
        pie_data = df_sorted.copy()
        pie_data['angle'] = pie_data[metric_type] / pie_data[metric_type].sum() * 2 * 3.14159
        pie_data['percentage'] = (pie_data[metric_type] / pie_data[metric_type].sum() * 100).round(1).astype(str) + '%'
        
        pie_chart = alt.Chart(pie_data).mark_arc().encode(
            theta='angle:Q',
            color=alt.Color('PC:N', scale=alt.Scale(scheme='tableau10')),
            tooltip=['PC', metric_type, 'percentage']
        ).properties(
            width=300,
            height=300
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
    # Key insights section
    st.subheader("Key Insights:")
    
    # Dynamically generate insights based on the data
    highest_tf = df_sorted.iloc[0]['PC']
    highest_tfidf_row = df_sorted.sort_values(by='TFIDF', ascending=False).iloc[0]
    highest_tfidf = highest_tfidf_row['PC']
    highest_tfidf_score = highest_tfidf_row['TFIDF']
    
    st.markdown(f"""
    - Within the **{selected_main_category}** category, **{highest_tf}** has the highest term frequency.
    - **{highest_tfidf}** has the highest TF-IDF score ({highest_tfidf_score:.2f}), indicating it's the most distinctive subcategory.
    - The distribution shows how each subcategory contributes to the overall {selected_main_category} category discussion.
    """)

# Function for Main Category Similarity Analysis page
def show_main_similarity():
    st.header("Main Category Similarity Analysis")
    
    if main_sim_df.empty:
        st.warning("Main category similarity data not available. Please make sure main_jaccard_matrix.csv is in the same directory as this app.")
        return
    
    # Convert similarity data to matrix format for visualization
    categories = sorted(main_sim_df['Category1'].unique())
    similarity_matrix = np.zeros((len(categories), len(categories)))
    
    # Fill the matrix with Jaccard values
    for _, row in main_sim_df.iterrows():
        i = categories.index(row['Category1'])
        j = categories.index(row['Category2'])
        similarity_matrix[i][j] = row['Jaccard']
    
    # Create a dataframe for the heatmap
    matrix_df = pd.DataFrame(similarity_matrix, index=categories, columns=categories)
    
    # Create a long-form dataframe for Altair
    heatmap_data = pd.DataFrame({
        'Category1': [cat1 for cat1 in categories for cat2 in categories],
        'Category2': [cat2 for cat1 in categories for cat2 in categories],
        'Jaccard': [matrix_df.loc[cat1, cat2] for cat1 in categories for cat2 in categories]
    })
    
    # Create heatmap
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('Category1:N', title='Category'),
        y=alt.Y('Category2:N', title='Category'),
        color=alt.Color('Jaccard:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Category1', 'Category2', 'Jaccard']
    ).properties(
        width=500,
        height=400,
        title="Jaccard Similarity Between Main Categories"
    )
    
    # Add text values
    text = alt.Chart(heatmap_data).mark_text().encode(
        x=alt.X('Category1:N'),
        y=alt.Y('Category2:N'),
        text=alt.Text('Jaccard:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Jaccard > 0.65,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    # Display heatmap with text
    st.altair_chart(heatmap + text, use_container_width=True)
    
    # Key insights section
    st.subheader("Key Insights:")
    
    # Find highest similarity pairs
    similarity_df_no_self = main_sim_df[main_sim_df['Category1'] != main_sim_df['Category2']]
    most_similar = similarity_df_no_self.sort_values(by='Jaccard', ascending=False).iloc[0]
    
    st.markdown(f"""
    - The Jaccard index measures similarity between categories based on document overlap.
    - **{most_similar['Category1']}** and **{most_similar['Category2']}** have the highest similarity ({most_similar['Jaccard']:.2f}).
    - Higher values (darker blue) indicate greater overlap in the documents where these categories appear.
    - This analysis helps identify which sustainability topics tend to be discussed together.
    """)

# Function for Sub Category Similarity Analysis page
def show_sub_similarity():
    st.header("Sub Category Similarity Analysis")
    
    if sub_sim_df.empty:
        st.warning("Sub-category similarity data not available. Please make sure sub_jaccard_matrix.csv is in the same directory as this app.")
        return
    
    # The sub-category similarity matrix is likely very large, so let's create a filtered view
    
    # Extract main categories from the Category1 column (format: MainCategory.SubCategory)
    sub_sim_df['MC1'] = sub_sim_df['Category1'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    sub_sim_df['MC2'] = sub_sim_df['Category2'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    
    # Extract sub categories
    sub_sim_df['SC1'] = sub_sim_df['Category1'].apply(lambda x: x.split('.')[1] if '.' in x else x)
    sub_sim_df['SC2'] = sub_sim_df['Category2'].apply(lambda x: x.split('.')[1] if '.' in x else x)
    
    # Main category filters
    st.subheader("Filter Similarity Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        main_cats = sorted(sub_sim_df['MC1'].unique())
        selected_mc1 = st.selectbox("Select First Main Category:", main_cats)
    
    with col2:
        main_cats2 = sorted(sub_sim_df['MC2'].unique())
        selected_mc2 = st.selectbox("Select Second Main Category:", main_cats2)
    
    # Filter based on selections
    filtered_sim = sub_sim_df[
        (sub_sim_df['MC1'] == selected_mc1) & 
        (sub_sim_df['MC2'] == selected_mc2)
    ]
    
    # Top similarities
    st.subheader("Top Subcategory Similarities")
    
    # Get top 10 similarities
    top_similarities = filtered_sim.sort_values(by='Jaccard', ascending=False).head(10)
    
    # Create a nice table
    st.table(top_similarities[['Category1', 'Category2', 'Jaccard']].reset_index(drop=True))
    
    # Create a heatmap visualization
    st.subheader("Similarity Heatmap")
    
    # Get unique subcategories
    subcats1 = sorted(filtered_sim['SC1'].unique())
    subcats2 = sorted(filtered_sim['SC2'].unique())
    
    # Create a matrix for heatmap
    heatmap_matrix = np.zeros((len(subcats1), len(subcats2)))
    
    # Fill the matrix
    for _, row in filtered_sim.iterrows():
        if row['SC1'] in subcats1 and row['SC2'] in subcats2:
            i = subcats1.index(row['SC1'])
            j = subcats2.index(row['SC2'])
            heatmap_matrix[i][j] = row['Jaccard']
    
    # Create a dataframe for the heatmap
    heatmap_df = pd.DataFrame(heatmap_matrix, index=subcats1, columns=subcats2)
    
    # Create a long-form dataframe for Altair
    heatmap_data = pd.DataFrame({
        'Subcategory1': [sc1 for sc1 in subcats1 for sc2 in subcats2],
        'Subcategory2': [sc2 for sc1 in subcats1 for sc2 in subcats2],
        'Jaccard': [heatmap_df.loc[sc1, sc2] for sc1 in subcats1 for sc2 in subcats2]
    })
    
    # Create heatmap
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('Subcategory2:N', title=f'{selected_mc2} Subcategories'),
        y=alt.Y('Subcategory1:N', title=f'{selected_mc1} Subcategories'),
        color=alt.Color('Jaccard:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Subcategory1', 'Subcategory2', 'Jaccard']
    ).properties(
        width=600,
        height=400,
        title=f"Jaccard Similarity Between {selected_mc1} and {selected_mc2} Subcategories"
    )
    
    # Add text values
    text = alt.Chart(heatmap_data).mark_text().encode(
        x=alt.X('Subcategory2:N'),
        y=alt.Y('Subcategory1:N'),
        text=alt.Text('Jaccard:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Jaccard > 0.4,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    # Display heatmap with text
    st.altair_chart(heatmap + text, use_container_width=True)
    
    # Key insights section
    st.subheader("Key Insights:")
    
    # Find the highest similarity pair
    if not filtered_sim.empty:
        most_similar = filtered_sim.sort_values(by='Jaccard', ascending=False).iloc[0]
        
        st.markdown(f"""
        - **{most_similar['Category1']}** and **{most_similar['Category2']}** have the highest similarity ({most_similar['Jaccard']:.2f}).
        - This suggests significant overlap in how these subcategories are discussed within the analyzed documents.
        - Higher similarity scores indicate topics that frequently appear together, potentially indicating conceptual relationships.
        - Sub-category similarities provide more granular insights than main category comparisons.
        """)
    else:
        st.warning("No similarity data available for the selected main categories.")

# Show the selected page
if page == "Main Categories":
    show_main_categories()
elif page == "Sub Categories":
    show_sub_categories()
elif page == "Main Category Similarity":
    show_main_similarity()
elif page == "Sub Category Similarity":
    show_sub_similarity()
