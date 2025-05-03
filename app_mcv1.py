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

# Function for Sub Categories page (no filters)
def show_sub_categories():
    st.header("Sub Categories Analysis - All Subcategories")
    
    if sub_df.empty:
        st.warning("Sub-category data not available. Please make sure sub_category_summary.csv is in the same directory as this app.")
        return
    
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
    df_sorted = sub_df.sort_values(by=metric_type, ascending=False).reset_index(drop=True)
    
    # Create a full-width chart
    st.subheader("All Sub-Categories Ranked")
    
    # Create combined category name for better labels
    df_sorted['Full_Category'] = df_sorted['MC'] + ': ' + df_sorted['PC']
    
    # Bar chart with Altair for all subcategories
    chart = alt.Chart(df_sorted).mark_bar().encode(
        y=alt.Y('Full_Category:N', sort=None, title=None),
        x=alt.X(f'{metric_type}:Q', title={
            "Total_Term_Frequency": "Total Term Frequency",
            "N_Cases": "Number of Cases",
            "TFIDF": "TF-IDF Score"
        }[metric_type]),
        color=alt.Color('MC:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Full_Category', metric_type, 'MC', 'PC']
    ).properties(
        height=600
    )
    
    st.altair_chart(chart, use_container_width=True)

    # Key insights section
    st.subheader("Key Insights:")
    
    # Calculate top subcategories for different metrics
    top_tf = df_sorted.iloc[0]['Full_Category']
    top_tfidf_row = df_sorted.sort_values(by='TFIDF', ascending=False).iloc[0]
    top_tfidf = top_tfidf_row['Full_Category']
    top_tfidf_score = top_tfidf_row['TFIDF']
    top_cases_row = df_sorted.sort_values(by='N_Cases', ascending=False).iloc[0]
    top_cases = top_cases_row['Full_Category']
    
    st.markdown(f"""
    - **{top_tf}** has the highest term frequency among all subcategories.
    - **{top_tfidf}** has the highest TF-IDF score ({top_tfidf_score:.2f}), indicating it's the most distinctive subcategory overall.
    - **{top_cases}** appears in the most cases, showing it has the widest coverage across documents.
    - The chart shows how subcategories from different main categories compare to each other.
    """)
    
    # Add grouped view by main category
    st.subheader("Sub-Categories by Main Category")
    
    # Create a multi-column view for main categories
    cols = st.columns(len(sub_df['MC'].unique()))
    
    # For each main category, show a bar chart of its subcategories
    for i, mc in enumerate(sorted(sub_df['MC'].unique())):
        with cols[i]:
            # Filter data for this main category
            mc_data = df_sorted[df_sorted['MC'] == mc].copy()
            
            # Sort by selected metric
            mc_data = mc_data.sort_values(by=metric_type, ascending=False)
            
            st.subheader(f"{mc}")
            
            # Create chart for this main category
            mc_chart = alt.Chart(mc_data).mark_bar().encode(
                y=alt.Y('PC:N', sort=None, title=None),
                x=alt.X(f'{metric_type}:Q', title=None),
                color=alt.Color('PC:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=['PC', metric_type]
            ).properties(
                height=200
            )
            
            st.altair_chart(mc_chart, use_container_width=True)
            
            # Show top subcategory for this main category
            top_subcat = mc_data.iloc[0]['PC']
            st.caption(f"Top subcategory: {top_subcat}")

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

# Function for Sub Category Similarity Analysis page (no filters)
def show_sub_similarity():
    st.header("Sub Category Similarity Analysis")
    
    if sub_sim_df.empty:
        st.warning("Sub-category similarity data not available. Please make sure sub_jaccard_matrix.csv is in the same directory as this app.")
        return
    
    # Process the sub-category data
    # Check the format of Category1 and Category2
    # They might be in format "MainCategory.SubCategory" or something else
    sample_cat = sub_sim_df['Category1'].iloc[0]
    
    if '.' in sample_cat:
        # Format is MainCategory.SubCategory
        sub_sim_df['MC1'] = sub_sim_df['Category1'].apply(lambda x: x.split('.')[0] if '.' in x else x)
        sub_sim_df['SC1'] = sub_sim_df['Category1'].apply(lambda x: x.split('.')[1] if '.' in x else x)
        sub_sim_df['MC2'] = sub_sim_df['Category2'].apply(lambda x: x.split('.')[0] if '.' in x else x)
        sub_sim_df['SC2'] = sub_sim_df['Category2'].apply(lambda x: x.split('.')[1] if '.' in x else x)
        
        # Create full display names
        sub_sim_df['Name1'] = sub_sim_df['MC1'] + ': ' + sub_sim_df['SC1']
        sub_sim_df['Name2'] = sub_sim_df['MC2'] + ': ' + sub_sim_df['SC2']
    else:
        # Assumed format is just subcategory names
        # In this case, we'll need to find a way to identify main categories
        # For now, just use the category names as is
        sub_sim_df['Name1'] = sub_sim_df['Category1']
        sub_sim_df['Name2'] = sub_sim_df['Category2']
    
    # Show top similarities
    st.subheader("Top 20 Subcategory Similarities")
    
    # Get top 20 similarities, excluding self-similarities
    top_similarities = sub_sim_df[sub_sim_df['Category1'] != sub_sim_df['Category2']]
    top_similarities = top_similarities.sort_values(by='Jaccard', ascending=False).head(20)
    
    # Create a bar chart of top similarities
    if 'Name1' in top_similarities.columns and 'Name2' in top_similarities.columns:
        top_similarities['Pair'] = top_similarities['Name1'] + ' & ' + top_similarities['Name2']
    else:
        top_similarities['Pair'] = top_similarities['Category1'] + ' & ' + top_similarities['Category2']
    
    top_chart = alt.Chart(top_similarities).mark_bar().encode(
        y=alt.Y('Pair:N', sort='-x', title=None),
        x=alt.X('Jaccard:Q', title='Jaccard Similarity'),
        color=alt.Color('Jaccard:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Pair', 'Jaccard']
    ).properties(
        height=600
    )
    
    st.altair_chart(top_chart, use_container_width=True)
    
    # Show heatmap of all similarities
    st.subheader("Similarity Heatmap for Top Subcategories")
    
    # Get the top subcategories for the heatmap (too many might make it unreadable)
    top_categories = set()
    for _, row in top_similarities.head(15).iterrows():
        top_categories.add(row['Category1'])
        top_categories.add(row['Category2'])
    
    top_categories = sorted(list(top_categories))
    
    # Filter the dataframe to only include these categories
    filtered_sim = sub_sim_df[
        (sub_sim_df['Category1'].isin(top_categories)) & 
        (sub_sim_df['Category2'].isin(top_categories))
    ]
    
    # Create a matrix for heatmap (assumes Category1 and Category2 contain the same values)
    heatmap_matrix = np.zeros((len(top_categories), len(top_categories)))
    
    # Fill the matrix
    for _, row in filtered_sim.iterrows():
        if row['Category1'] in top_categories and row['Category2'] in top_categories:
            i = top_categories.index(row['Category1'])
            j = top_categories.index(row['Category2'])
            heatmap_matrix[i][j] = row['Jaccard']
    
    # Create a dataframe for the heatmap
    display_categories = []
    for cat in top_categories:
        if 'Name1' in sub_sim_df.columns:
            # Find the display name for this category
            display_name = sub_sim_df[sub_sim_df['Category1'] == cat]['Name1'].iloc[0]
            display_categories.append(display_name)
        else:
            display_categories.append(cat)
    
    heatmap_df = pd.DataFrame(heatmap_matrix, index=display_categories, columns=display_categories)
    
    # Create a long-form dataframe for Altair
    heatmap_data = pd.DataFrame({
        'Category1': [cat1 for cat1 in display_categories for cat2 in display_categories],
        'Category2': [cat2 for cat1 in display_categories for cat2 in display_categories],
        'Jaccard': [heatmap_df.loc[cat1, cat2] for cat1 in display_categories for cat2 in display_categories]
    })
    
    # Create heatmap
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('Category2:N', title=None),
        y=alt.Y('Category1:N', title=None),
        color=alt.Color('Jaccard:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Category1', 'Category2', 'Jaccard']
    ).properties(
        width=700,
        height=600,
        title="Jaccard Similarity Between Top Subcategories"
    )
    
    # Add text values
    text = alt.Chart(heatmap_data).mark_text().encode(
        x=alt.X('Category2:N'),
        y=alt.Y('Category1:N'),
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
    if not top_similarities.empty:
        most_similar = top_similarities.iloc[0]
        pair_name = most_similar['Pair'] if 'Pair' in most_similar else f"{most_similar['Category1']} & {most_similar['Category2']}"
        
        st.markdown(f"""
        - **{pair_name}** have the highest similarity ({most_similar['Jaccard']:.2f}).
        - This suggests significant overlap in how these subcategories are discussed within the analyzed documents.
        - Higher similarity scores indicate topics that frequently appear together, potentially indicating conceptual relationships.
        - Sub-category similarities provide more granular insights than main category comparisons.
        """)
    else:
        st.warning("No similarity data available for analysis.")

# Show the selected page
if page == "Main Categories":
    show_main_categories()
elif page == "Sub Categories":
    show_sub_categories()
elif page == "Main Category Similarity":
    show_main_similarity()
elif page == "Sub Category Similarity":
    show_sub_similarity()
