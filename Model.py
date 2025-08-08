import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans # Suitable for large datasets
from sklearn.metrics import silhouette_score # Useful for evaluating cluster quality
import matplotlib.pyplot as plt # For plotting, e.g., Elbow method
import seaborn as sns # For enhanced visualizations

# --- NLTK Data Downloads ---
# Ensure required NLTK data is downloaded. Use try-except to avoid re-downloading.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Data Loading Function ---
def read_large_csv(file_path, columns, chunk_size=50000):
    """
    Reads a large CSV file in chunks, assigns custom column names, and handles
    potential encoding issues.

    Args:
        file_path (str): The path to the CSV file.
        columns (list): A list of column names to assign to the DataFrame.
        chunk_size (int): The number of rows to read at a time.

    Returns:
        pandas.DataFrame: The concatenated DataFrame from all chunks.
    """
    print("📥 Loading CSV file in chunks...")
    df_list = []
    
    # Attempt to read with 'utf-8', fallback to 'latin1' if UnicodeDecodeError occurs
    encodings_to_try = ['utf-8', 'latin1']
    for encoding in encodings_to_try:
        try:
            for i, chunk in enumerate(pd.read_csv(file_path, header=None, chunksize=chunk_size, encoding=encoding, low_memory=False)):
                # Check if the number of columns in the chunk matches the expected number
                if chunk.shape[1] == len(columns):
                    chunk.columns = columns
                    df_list.append(chunk)
                else:
                    print(f"⚠️ Skipping chunk {i+1} due to column count mismatch. Expected {len(columns)}, got {chunk.shape[1]}.")
            print(f"CSV loading complete with encoding '{encoding}'!")
            return pd.concat(df_list, ignore_index=True) # Successfully loaded, return DataFrame
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError with encoding '{encoding}'. Trying next encoding...")
            df_list = [] # Clear df_list for the next encoding attempt
        except Exception as e:
            print(f"An error occurred while reading with encoding '{encoding}': {e}")
            df_list = [] # Clear df_list for the next encoding attempt

    print("Failed to load data with all attempted encodings. Check file path or data integrity.")
    return pd.DataFrame(columns=columns) # Return an empty DataFrame with correct columns if all attempts fail

# --- Text Cleaning Functions ---
def clean_text(text):
    """
    Cleans a single string by converting to lowercase, removing URLs,
    punctuation, numbers, and extra spaces.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text string.
    """
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def preprocess_text(df, text_column):
    """
    Preprocesses the text column of a DataFrame by cleaning, tokenizing,
    and removing stopwords. Handles NaN values gracefully.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text data.

    Returns:
        list: A list of preprocessed text strings.
    """
    print(f"🧹 Preprocessing text in column '{text_column}'...")
    stop_words = set(stopwords.words('english'))
    
    # Apply initial cleaning and fill NaN values before tokenization
    # This ensures all entries are strings and are initially cleaned
    df[text_column] = df[text_column].fillna('').apply(clean_text)

    # Use a list comprehension for potentially faster processing than a traditional for loop
    cleaned_texts = [
        " ".join([word for word in word_tokenize(sentence) if word not in stop_words])
        if sentence else ""
        for sentence in df[text_column]
    ]
    
    print(" Text preprocessing complete!")
    return cleaned_texts

# --- AI Website Identification Function ---
def identify_ai_websites(df, text_column):
    """
    Identifies websites that talk about AI based on a predefined list of keywords.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing preprocessed text.

    Returns:
        pandas.DataFrame: The DataFrame with a new 'is_ai_related' boolean column.
    """
    print(" Identifying AI-related websites...")
    ai_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'robotics', 'natural language processing', 'nlp',
        'computer vision', 'data science', 'predictive analytics', 'automation',
        'generative ai', 'llm', 'large language model', 'gpt', 'transformer'
    ]
    # Create a regex pattern to match any of the keywords as whole words (\b for word boundary)
    pattern = r'\b(' + '|'.join(re.escape(k) for k in ai_keywords) + r')\b'
    
    # Ensure the text column is string type before applying str.contains
    # This is important as it might contain NaN or other non-string types if not handled correctly upstream
    df[text_column] = df[text_column].astype(str)
    
    # Check if any keyword is present in the cleaned text (case-insensitive)
    df['is_ai_related'] = df[text_column].str.contains(pattern, case=False, na=False)
    print(" AI website identification complete!")
    return df

# --- Website Clustering Function ---
def cluster_websites(df, text_column, n_clusters=10, max_features=5000):
    """
    Clusters websites based on their text content using TF-IDF vectorization
    and MiniBatchKMeans.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing preprocessed text.
        n_clusters (int): The number of clusters to form.
        max_features (int): The maximum number of features (vocabulary size) for TF-IDF.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The DataFrame with a new 'cluster_label' column.
            - TfidfVectorizer: The fitted TF-IDF vectorizer.
            - MiniBatchKMeans: The fitted MiniBatchKMeans model.
    """
    print(f"Vectorizing text and clustering into {n_clusters} clusters...")
    
    # Filter out empty strings before vectorization, as TF-IDF cannot process them
    non_empty_texts_series = df[df[text_column].str.strip() != ''][text_column]
    
    if non_empty_texts_series.empty:
        print(" No non-empty text found for vectorization and clustering. All rows will be unclustered (-1).")
        df['cluster_label'] = -1 # Assign a default label for no clusters
        return df, None, None

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(non_empty_texts_series)
    
    # Initialize cluster labels for all rows with -1 (unclustered/not yet assigned)
    df['cluster_label'] = -1

    # Train MiniBatchKMeans model
    # n_init='auto' is the recommended default for scikit-learn >= 1.0
    # verbose=0 suppresses training output
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
    cluster_labels = model.fit_predict(X)
    
    # Assign cluster labels back to the original DataFrame, only for the rows that were clustered
    df.loc[non_empty_texts_series.index, 'cluster_label'] = cluster_labels
    
    print("Website clustering complete!")
    return df, vectorizer, model

# --- Cluster Analysis Function ---
def analyze_clusters(df, vectorizer, kmeans_model): # Added kmeans_model as an argument
    """
    Analyzes the formed clusters by calculating the proportion of AI-related
    websites in each cluster and identifying the top keywords for each cluster.

    Args:
        df (pandas.DataFrame): The DataFrame with 'is_ai_related' and 'cluster_label' columns.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        kmeans_model (MiniBatchKMeans): The fitted MiniBatchKMeans model.
    """
    print("Analyzing clusters...")
    
    if 'cluster_label' not in df.columns or df['cluster_label'].isnull().all() or df['cluster_label'].nunique() <= 1:
        print("No meaningful clusters to analyze (or only one cluster).")
        return

    # Filter out unclustered data (-1 label) for analysis
    clustered_df = df[df['cluster_label'] != -1]
    if clustered_df.empty:
        print("No clustered data to analyze.")
        return

    # Proportion of AI-related websites per cluster
    # Using .mean() on boolean (True/False) column gives the proportion of True values
    cluster_ai_proportion = clustered_df.groupby('cluster_label')['is_ai_related'].mean().reset_index()
    cluster_ai_proportion.rename(columns={'is_ai_related': 'ai_proportion'}, inplace=True)
    print("\nProportion of AI-related websites per cluster:")
    print(cluster_ai_proportion.to_string(index=False)) # Use to_string for better console output

    # Top keywords for each cluster - OPTIMIZED to use cluster_centers_
    print("\nTop keywords for each cluster (to understand cluster themes):")
    terms = vectorizer.get_feature_names_out()
    
    # Use the cluster centers directly from the KMeans model
    # These represent the average TF-IDF vector for each cluster
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1] # Sort in descending order

    for i in sorted(clustered_df['cluster_label'].unique()):
        print(f"Cluster {i}:")
        # Get the top 10 terms for the current cluster
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"  Top terms: {', '.join(top_terms)}")

    print("Cluster analysis complete!")

# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the data loading, preprocessing, AI identification,
    clustering, and analysis steps.
    """
    # --- IMPORTANT ---
    # For a 3GB Excel file, it is CRUCIAL to convert it to CSV first.
    # pandas.read_excel does not support chunking directly, which can lead to
    # out-of-memory errors for very large files.
    # You can typically do this by opening the Excel file in a spreadsheet program
    # (like Microsoft Excel, Google Sheets, LibreOffice Calc) and saving it as a CSV.
    file_path = r"C:\Users\Admin\Downloads\Dissertation project\Model\Code\df202350.csv" 
    
    # Define your column names in the order they appear in your CSV file
    required_columns = ['urls', 'domains', 'postcodes', 'data_source', 'text']

    # Step 1: Read CSV
    df = read_large_csv(file_path, required_columns)
    
    if df.empty:
        print("Exiting: DataFrame is empty after loading. Please check your file path and content.")
        return

    # Step 2: Preprocess text
    # This will create a new 'cleaned_text' column in the DataFrame
    df['cleaned_text'] = preprocess_text(df, 'text')

    # Step 3: Drop rows where 'cleaned_text' is empty or NaN after preprocessing
    # This ensures only meaningful text is used for subsequent analysis
    initial_rows = len(df)
    # Replace empty strings with NaN for proper dropna functionality
    df['cleaned_text'].replace('', np.nan, inplace=True) 
    df.dropna(subset=['cleaned_text'], inplace=True)
    rows_after_cleaning = len(df)
    print(f"Removed {initial_rows - rows_after_cleaning} rows with empty or non-meaningful text after cleaning.")

    if df.empty:
        print("All meaningful rows were removed after preprocessing. Check your data and preprocessing steps.")
        return

    # Step 4: Identify AI-related websites
    # This adds an 'is_ai_related' boolean column to the DataFrame
    df = identify_ai_websites(df, 'cleaned_text')

    # Step 5: Cluster websites based on content
    # You might need to experiment with 'n_clusters' to find the optimal number
    # for your dataset. Methods like the Elbow method or Silhouette score can help.
    # For a starting point, we'll use 10 clusters.
    n_clusters_to_try = 10 
    df, vectorizer, kmeans_model = cluster_websites(df, 'cleaned_text', n_clusters=n_clusters_to_try)

    if df.empty or 'cluster_label' not in df.columns or df['cluster_label'].isnull().all():
        print("Exiting: Clustering could not be performed or resulted in an empty DataFrame/no clusters.")
        return


    # Step 6: Analyze clusters to differentiate AI content
    analyze_clusters(df, vectorizer, kmeans_model)

# Step 7: Save Results
    save_cluster_results(df)

    print("\n--- Model Execution Summary ---")
    print("The DataFrame 'df' now contains the following new columns:")
    print("  - 'cleaned_text': Your preprocessed website content.")
    print("  - 'is_ai_related': A boolean flag indicating if the website talks about AI.")
    print("  - 'cluster_label': The assigned cluster ID for each website.")
    print("\n Output files have been saved. You can now open them for review or visualization.")

def save_cluster_results(df, output_dir='outputs'): 
    """
    Saves the full clustered DataFrame, AI-related websites, and non-AI websites into separate CSV files.

    Args:
        df (pandas.DataFrame): The DataFrame with clustering and AI-related info.
        output_dir (str): Folder to store the outputs.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    full_output_path = os.path.join(output_dir, 'clustered_websites_all.csv')
    ai_output_path = os.path.join(output_dir, 'ai_websites.csv')
    non_ai_output_path = os.path.join(output_dir, 'non_ai_websites.csv')

    # Save full DataFrame
    df.to_csv(full_output_path, index=False)
    print(f"Full clustered data saved to: {full_output_path}")

    # Save AI-related websites
    df[df['is_ai_related']].to_csv(ai_output_path, index=False)
    print(f" AI-related websites saved to: {ai_output_path}")

    # Save non-AI-related websites
    df[~df['is_ai_related']].to_csv(non_ai_output_path, index=False)
    print(f" Non-AI websites saved to: {non_ai_output_path}")


if __name__ == "__main__":
    main()
