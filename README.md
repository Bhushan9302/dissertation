# 📍 Mapping the Spatial Diffusion of AI in the UK

This dissertation project implements an advanced **machine learning and spatial analysis pipeline** to model the adoption and diffusion of **Artificial Intelligence (AI)** across the United Kingdom.

Moving beyond binary classification, this solution integrates:

- **Unsupervised Topic Modeling (LDA)**
- **Spatial Econometrics**
- **Entity Resolution Techniques**
- **Urban Hierarchy Theory**

The objective is to empirically test geographic theories of innovation diffusion using high-fidelity firm-level data.

---

## 🚀 Key Features

### 🏢 Firm-Level Entity Resolution
Aggregates sub-pages into **Base Domains** to prevent overcounting and ensure each record represents a unique corporate entity.

### 📍 Headquarters Proxy Modeling
Implements a **frequency-based postcode proxy (Mode)** to statistically isolate the primary operational location from multiple scraped postcodes.

### 🧠 Sectoral Discovery (LDA)
Uses **Latent Dirichlet Allocation (LDA)** to uncover five distinct AI sub-sectors:

- Management  
- Commerce  
- Engineering  
- Infrastructure  
- Automation  

This confirms AI as a **General Purpose Technology (GPT)**.

### ⚙️ Scalable NLP Pipeline
Processes extremely large textual datasets using:

- Pandas chunking  
- Advanced tokenization  
- Efficient memory management  

### 🗺️ Spatial Unit Optimization
Aggregates location data to the **UK Outcode (Postal District)** level to:

- Reduce geocoding noise  
- Preserve city-level analytical resolution  

### 📊 Analytical Mapping
Generates **Tableau-ready datasets** to visualize:

- Hierarchical Diffusion  
- Agglomeration Effects  
- Urban Hierarchy Patterns  

### 📈 Specialization Analysis
Provides framework for calculating **Location Quotients (LQ)** to identify AI specialization hotspots.

---

## 🛠️ Prerequisites

To run this project, you will need:

- Python 3.8+
- `pandas`
- `numpy`
- `nltk`
- `gensim`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- Tableau Desktop or Tableau Public (for spatial visualization)

⚠️ The large dataset (`ai_websites.csv`) must be present in the local directory or accessible via S3.

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Bhushan9302/dissertation.git
cd dissertation
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy nltk gensim scikit-learn matplotlib seaborn
```

### 3️⃣ Download NLTK Data

The script automatically downloads required NLTK packages:

- `stopwords`
- `punkt`

---

## ⚙️ Configuration

Primary configuration is handled inside the `run_topic_modeling()` function.

| Variable      | Description                                           | Default Value                         |
|--------------|-------------------------------------------------------|---------------------------------------|
| INPUT_FILE   | Path to raw scraped CSV dataset                       | `"ai_websites.csv"`                   |
| num_topics   | Number of sub-sectors to discover via LDA             | `5`                                   |
| passes       | Number of training passes for the LDA model           | `5`                                   |
| chunk_size   | Rows processed per memory cycle                       | `1500`                                |
| OUTPUT_FILE  | Final CSV optimized for Tableau mapping               | `"ai_firms_with_topics.csv"`          |

---

## ▶️ How to Run the Script

Execute the topic modeling pipeline from your terminal:

```bash
python Topic_Modeling.py
```

---

## 📚 Theoretical Frameworks Tested

This project empirically evaluates three core academic pillars:

### 1️⃣ Hierarchical Diffusion
Tests whether AI adoption "jumps" from major hubs (e.g., London) to secondary cities.

### 2️⃣ Epidemic Effects
Maps geographic clustering formed through proximity and knowledge spillovers.

### 3️⃣ GPT Pervasiveness
Demonstrates AI's penetration across diverse and unrelated economic sectors.

---

## 🏙️ Research Contribution

This dissertation bridges:

- Machine Learning  
- Economic Geography  
- Innovation Theory  
- Spatial Econometrics  

It delivers a **replicable, scalable, and theory-driven framework** for analyzing technological diffusion at firm-level spatial resolution.

---

## 📌 Output

The final output file:

```
ai_firms_with_topics.csv
```

is fully optimized for:

- Tableau visualization  
- Spatial econometric modeling  
- Location Quotient analysis  
- Urban hierarchy mapping  

---

## 📜 License

This project is for academic research purposes.  
Please cite appropriately if used in further research.

---

## 👤 Author

Bhushan  

MSc Dissertation Project  
Mapping the Spatial Diffusion of AI in the UK  
