import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import os
import gc  
from collections import Counter
from urllib.parse import urlparse
import ast

# 1. SETUP PATHS
INPUT_FILE = r'D:\Dissertation project\Model\Code\model_outputs\ai_websites.csv'
OUTPUT_FILE = r'D:\Dissertation project\Model\Code\model_outputs\ai_firms_with_topics.csv'

def get_base_domain(url_string):
    """Extracts the base domain (e.g., hl.co.uk) to group sub-pages together."""
    try:
        domain = urlparse(str(url_string)).netloc
        domain = domain.replace('www.', '')
        return domain if domain else str(url_string)
    except:
        return str(url_string)

def run_topic_modeling():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: The file {INPUT_FILE} was not found.")
        return

    print("--- Phase 1: Aggregating Sub-Pages into Unique Firms ---")
    text_collector = {}
    postcode_collector = {}

    chunk_size = 1500
    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, low_memory=False):
        for _, row in chunk.iterrows():
            # Get the base domain instead of the full sub-page URL
            domain = get_base_domain(row.get('urls', 'unknown'))
            text = str(row.get('text', ''))
            pc_raw = str(row.get('postcodes', 'unknown'))

            if domain not in postcode_collector:
                postcode_collector[domain] = [pc_raw]
            else:
                postcode_collector[domain].append(pc_raw)

            # Keep text short to prevent memory crashes
            if domain not in text_collector:
                text_collector[domain] = text[:6000]
            else:
                if len(text_collector[domain]) < 6000:
                    text_collector[domain] += " " + text[:2000]
        
    print(f"Total Unique AI Firms identified: {len(text_collector)}")

    print("--- Phase 2: Identifying True Headquarters ---")
    master_data = []
    for domain in text_collector:
        # Pool all postcodes found across all sub-pages of this domain
        all_postcodes = []
        for pc_string in postcode_collector[domain]:
            # Clean out Python brackets from the CSV strings
            cleaned_str = pc_string.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
            # Split by comma if there are multiple in one string
            pieces = [p.strip() for p in cleaned_str.split(',') if p.strip()]
            all_postcodes.extend(pieces)
        
        # Mathematical Proof of HQ: Find the Mode (most frequent)
        if all_postcodes:
            true_hq = Counter(all_postcodes).most_common(1)[0][0]
        else:
            true_hq = "Unknown"

        master_data.append({
            'domain': domain, 
            'postcodes': true_hq, 
            'text': text_collector[domain]
        })

    master_df = pd.DataFrame(master_data)
    
    del text_collector
    del postcode_collector
    gc.collect() 

    print("--- Phase 3: Text Preprocessing ---")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['company', 'firm', 'business', 'service', 'uk', 'solution', 'management', 'provide', 'view', 'page', 'click', 'information'])
    
    tokenizer = RegexpTokenizer(r'\w+')

    def preprocess(val):
        tokens = tokenizer.tokenize(str(val).lower())
        return [t for t in tokens if t not in stop_words and len(t) > 3]

    processed_docs = master_df['text'].apply(preprocess)

    print("--- Phase 4: Running LDA (Sector Discovery) ---")
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.4)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=5, random_state=42)

    master_df['topic_id'] = [max(lda_model[c], key=lambda x: x[1])[0] for c in corpus]
    
    topic_labels = {
        0: "Management",
        1: "Commerce",
        2: "Engineering",
        3: "Infrastructure",
        4: "Automation"
    }
    master_df['sector_name'] = master_df['topic_id'].map(topic_labels)

    # 6. EXPORT (Tableau Ready!)
    final_output = master_df[['domain', 'postcodes', 'topic_id', 'sector_name']]
    final_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS! High-Quality Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_topic_modeling()