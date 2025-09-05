import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from collections import Counter
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class WebsiteAIClassifier:
    def __init__(self, sample_size=10000, test_size=0.2, random_state=42, max_features=5000):
        """ Initialize the Website AI Classifier for model training and validation """
        self.sample_size = sample_size
        # ... (all the other self.variable assignments) ...

    # Add this line to call the downloader
        self._download_nltk_data()
        
        # Enhanced AI keywords
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'neural network', 'neural networks', 'natural language processing', 'nlp',
            'computer vision', 'data science', 'predictive analytics', 'automation',
            'generative ai', 'large language model', 'llm', 'gpt', 'bert', 'transformer',
            'chatbot', 'conversational ai', 'recommendation system', 'autonomous',
            'tensorflow', 'pytorch', 'scikit-learn', 'openai', 'anthropic', 'hugging face'
        ]
        
        self._download_nltk_data()

    def _download_nltk_data(self):
    """Checks for all required NLTK data and downloads any that are missing."""
    # A dictionary to manage all required packages and their types
    required_packages = {
        'corpora': ['stopwords'],
        'tokenizers': ['punkt', 'punkt_tab']
    }
    
    # Loop through the packages and download if missing
    for resource_type, packages in required_packages.items():
        for package in packages:
            try:
                # Check if the resource is available at its correct path
                nltk.data.find(f'{resource_type}/{package}')
            except LookupError:
                print(f"NLTK package '{package}' not found. Downloading...")
                # Download the package quietly to keep the output clean
                nltk.download(package, quiet=True)

    def load_sample_data(self, file_path, columns):
        """Load a sample of data with improved memory management"""
        print(f"Loading sample of {self.sample_size:,} rows...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        encodings_to_try = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings_to_try:
            try:
                print(f"Trying encoding: {encoding}")
                chunk_reader = pd.read_csv(
                    file_path, header=None, chunksize=5000, encoding=encoding,
                    low_memory=False, on_bad_lines='skip'
                )
                
                sampled_data = []
                total_processed = 0
                for chunk in chunk_reader:
                    for _, row in chunk.iterrows():
                        total_processed += 1
                        if len(sampled_data) < self.sample_size:
                            sampled_data.append(row)
                        else:
                            replace_idx = np.random.randint(0, total_processed)
                            if replace_idx < self.sample_size:
                                sampled_data[replace_idx] = row
                    if len(sampled_data) >= self.sample_size and total_processed > self.sample_size * 2:
                        break

                if sampled_data:
                    df = pd.DataFrame([list(row) for row in sampled_data])
                    num_cols = min(len(columns), df.shape[1])
                    df = df.iloc[:, :num_cols]
                    df.columns = columns[:num_cols]
                    print(f"Successfully loaded {len(df):,} rows.")
                    return df
            except Exception as e:
                print(f"Failed with encoding '{encoding}': {e}")
        raise Exception("All loading methods failed. Please check your CSV file format.")

    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+|<[^>]+>|\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_text(self, df, text_column):
        """Preprocess text data"""
        print(f"Preprocessing text in column '{text_column}'...")
        stop_words = set(stopwords.words('english'))
        df['cleaned_text'] = df[text_column].fillna('').apply(self.clean_text)
        
        processed_texts = []
        for text in df['cleaned_text']:
            if text and len(text.split()) > 2:
                tokens = word_tokenize(text)
                filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
                processed_texts.append(" ".join(filtered))
            else:
                processed_texts.append("")
        df['cleaned_text'] = processed_texts
        print("Text preprocessing complete.")
        return df

    def identify_ai_websites(self, df, text_column):
        """Identify AI-related websites and track keywords"""
        print("Identifying AI websites and finding keywords...")
        
        ai_classifications = []
        found_keywords_list = []
        
        # Create a regex pattern for efficiency
        keyword_pattern = re.compile(r'\b(' + '|'.join(self.ai_keywords) + r')\b')

        for text in df[text_column].astype(str):
            matches = set(keyword_pattern.findall(text.lower()))
            if matches:
                ai_classifications.append(True)
                found_keywords_list.append(", ".join(sorted(list(matches))))
            else:
                ai_classifications.append(False)
                found_keywords_list.append("")

        df['is_ai_related'] = ai_classifications
        df['found_ai_keywords'] = found_keywords_list
        
        ai_count = df['is_ai_related'].sum()
        print(f"Found {ai_count:,} AI-related websites ({ai_count/len(df):.2%}).")
        return df

    def create_balanced_train_test_split(self, df, text_column, target_column):
        """Create balanced training and testing splits"""
        print("Creating balanced train-test split...")
        df_clean = df[df[text_column].str.len() > 10].copy()
        X, y = df_clean[text_column], df_clean[target_column]
        
        if y.nunique() < 2:
            print("Warning: Only one class present. Cannot create a stratified split. Using regular split.")
            return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Training set: {len(X_train):,} | Testing set: {len(X_test):,}")
        return X_train, X_test, y_train, y_test

    def vectorize_text(self, X_train, X_test):
        """Vectorize text data using TF-IDF"""
        print("Starting TF-IDF Vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, min_df=3, max_df=0.8,
            ngram_range=(1, 2), sublinear_tf=True
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"Vectorization complete. Vocabulary size: {len(self.vectorizer.get_feature_names_out()):,}")
        return X_train_vec, X_test_vec

    def train_classifier(self, X_train_vec, y_train):
        """Train AI classification model"""
        print("Training classification model...")
        self.classifier_model = LogisticRegression(
            random_state=self.random_state, class_weight='balanced', solver='liblinear'
        )
        self.classifier_model.fit(X_train_vec, y_train)
        print("Model training complete.")

    def evaluate_model(self, X_test_vec, y_test):
        """Evaluate model performance and show accuracy"""
        print("\n" + "="*25)
        print("MODEL EVALUATION RESULTS")
        print("="*25)
        
        y_pred = self.classifier_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nACCURACY: {accuracy:.2%}")
        
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Non-AI', 'AI']))
        
        self._create_evaluation_plots(y_test, y_pred)
        
        return {'accuracy': accuracy, 'report': classification_report(y_test, y_pred, output_dict=True)}

    def _create_evaluation_plots(self, y_test, y_pred):
        """Create and save evaluation plots"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-AI', 'AI'], yticklabels=['Non-AI', 'AI'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.subplot(1, 2, 2)
        pd.Series(y_test).value_counts().sort_index().plot(kind='bar', color='skyblue', position=0, width=0.4, label='Actual')
        pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', color='salmon', position=1, width=0.4, label='Predicted')
        plt.title('Actual vs. Predicted Distribution')
        plt.xticks(ticks=[0, 1], labels=['Non-AI', 'AI'], rotation=0)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300)
        print("\nEvaluation plots saved as 'model_evaluation.png'")
        plt.show()

    # --- UPDATED FUNCTION ---
    def save_results(self, df, metrics, output_dir='model_outputs'):
        """Save comprehensive results including all datasets and model artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete dataset
        complete_path = os.path.join(output_dir, 'complete_dataset.csv')
        df.to_csv(complete_path, index=False)
        print(f"Complete dataset saved to: {complete_path}")

        # Save the AI-related websites
        ai_path = os.path.join(output_dir, 'ai_websites.csv')
        df[df['is_ai_related']].to_csv(ai_path, index=False)
        print(f"AI websites dataset saved to: {ai_path}")
        
        # Save the non-AI-related websites
        non_ai_path = os.path.join(output_dir, 'non_ai_websites.csv')
        df[~df['is_ai_related']].to_csv(non_ai_path, index=False)
        print(f"Non-AI websites dataset saved to: {non_ai_path}")
        
        # Save model artifacts
        joblib.dump(self.classifier_model, os.path.join(output_dir, 'classifier_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        
        # Save performance report
        import json
        with open(os.path.join(output_dir, 'performance_report.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model and report saved to '{output_dir}' folder.")


def main():
    """Main function updated to download from S3 and process in chunks."""
    print("Website AI Classification Workflow Initialized")
    print("=" * 50)
    
    # Note: sample_size is removed to process the whole file
    classifier = WebsiteAIClassifier(test_size=0.25, max_features=8000)
    
    # --- PATHS ON THE EC2 INSTANCE ---
    s3_bucket = "bhushan-dissertation-data-2025" # Your S3 bucket name
    s3_filename = "df202350.csv" # The name of the file in your S3 bucket
    local_filepath = s3_filename # Use the same name for the local file
    output_dir = "model_outputs"
    required_columns = ['urls', 'domains', 'postcodes', 'data_source', 'text']
    
    try:
        # --- STEP 1: DOWNLOAD DATA FROM S3 ---
        s3_path = f"s3://{s3_bucket}/{s3_filename}"
        print(f"Downloading data from {s3_path} to the EC2 instance...")
        # This command runs in the terminal to copy the file
        os.system(f"aws s3 cp {s3_path} .")
        print("Download complete.")

        # --- STEP 2: PROCESS THE LOCAL CSV IN CHUNKS ---
        chunk_reader = pd.read_csv(
            local_filepath, header=None, chunksize=50000,
            low_memory=False, on_bad_lines='skip', names=required_columns
        )
        
        all_results_df = pd.DataFrame()
        
        print("Starting to process the dataset in chunks...")
        for i, chunk_df in enumerate(chunk_reader):
            print(f"--- Processing chunk {i+1} ---")
            
            processed_chunk = classifier.preprocess_text(chunk_df, 'text')
            processed_chunk = processed_chunk[processed_chunk['cleaned_text'].str.strip() != ''].copy()
            identified_chunk = classifier.identify_ai_websites(processed_chunk, 'cleaned_text')
            all_results_df = pd.concat([all_results_df, identified_chunk], ignore_index=True)
            
        print("\nAll chunks processed successfully!")
        
        df = all_results_df

        # --- The rest of the script continues as before ---
        X_train, X_test, y_train, y_test = classifier.create_balanced_train_test_split(df, 'cleaned_text', 'is_ai_related')
        X_train_vec, X_test_vec = classifier.vectorize_text(X_train, X_test)
        classifier.train_classifier(X_train_vec, y_train)
        metrics = classifier.evaluate_model(X_test_vec, y_test)
        classifier.save_results(df, metrics, output_dir=output_dir)
        
        # --- STEP 3: UPLOAD RESULTS BACK TO S3 ---
        print(f"Uploading results to s3://{s3_bucket}/results/...")
        os.system(f"aws s3 cp {output_dir} s3://{s3_bucket}/results/ --recursive")

    except FileNotFoundError:
        print(f"ERROR: Could not find the file '{local_filepath}'. Did the download from S3 fail?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        gc.collect()
        print("\nMemory cleaned up.")

if __name__ == "__main__":
    main()