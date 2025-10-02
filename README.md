# dissertation
# Website AI Classifier

This project implements a complete machine learning pipeline to classify websites based on their content, specifically identifying sites related to **Artificial Intelligence (AI)**, **Machine Learning (ML)**, and **Deep Learning (DL)**.

The solution is optimized for handling large datasets by utilizing a **chunking** and **stratified sampling** strategy, and includes a full workflow for data acquisition from **AWS S3** and results persistence back to S3.

## 🚀 Key Features

* **Robust Data Acquisition:** Downloads the raw dataset directly from a specified AWS S3 bucket.
* **Scalable Processing:** Uses Pandas **chunking** to process extremely large CSV files without overwhelming memory.
* **AI Keyword Identification:** Initial classification based on an extensive list of AI/ML keywords (e.g., `LLM`, `GPT`, `TensorFlow`).
* **Text Preprocessing:** Full NLP pipeline including cleaning, tokenization, and stop-word removal.
* **Feature Engineering:** Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** for effective text vectorization.
* **Stratified Training:** Employs `StratifiedShuffleSplit` to ensure balanced class representation during model training.
* **Model Persistence:** Saves the trained **Logistic Regression** model and the **TF-IDF Vectorizer** for future inference.
* **Visual Evaluation:** Generates a **Confusion Matrix** and a class distribution plot to visualize performance.
* **Cloud Persistence:** Uploads all results (models, datasets, and performance reports) back to S3.

## 🛠️ Prerequisites

To run this script, you will need the following:

1.  **Python 3.8+**
2.  An **AWS EC2 Instance** (Recommended for memory handling)
3.  **AWS CLI** configured on the EC2 instance with access to the specified S3 bucket.
4.  The large dataset (`df202350.csv`) must be present in the S3 bucket.

## 📦 Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Bhushan9302/dissertation.git](https://github.com/Bhushan9302/dissertation.git)
    cd dissertation
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib
    ```

3.  **Download NLTK Data:**
    The script will automatically attempt to download the necessary NLTK packages (`stopwords` and `punkt`).

## ⚙️ Configuration

The primary configuration is done within the `main()` function:

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `s3_bucket` | Your AWS S3 bucket name. **Must be updated.** | `"bhushan-dissertation-data-2025"` |
| `s3_filename` | The name of the input CSV file in your S3 bucket. | `"df202350.csv"` |
| `test_size` | The proportion of data reserved for testing. | `0.25` |
| `max_features` | The max vocabulary size for the TF-IDF vectorizer. | `8000` |
| `output_dir` | Local folder where all outputs will be saved before S3 upload. | `"model_outputs"` |

## ▶️ How to Run the Script

Run the main Python file from your EC2 terminal:

```bash
python model.py
