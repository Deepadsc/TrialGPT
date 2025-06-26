# Patient Clinical Trial Matching (PCTM) System

An advanced AI-powered system designed to match patient profiles with suitable clinical trials using a hybrid retrieval and ranking approach. The system combines traditional information retrieval techniques with modern language models to provide accurate and relevant clinical trial matches.

## 🌟 Features

- **Hybrid Retrieval**: Combines BM25 and dense retrieval for comprehensive search
- **Advanced Matching**: Semantic matching of patient profiles to trial criteria
- **Intelligent Ranking**: Learning-to-rank techniques for optimal trial prioritization
- **Query Understanding**: Automatic keyword generation for better search results
- **Modular Architecture**: Separated retrieval, matching, and ranking components
- **Command-line Interface**: Easy-to-use interactive interface
- **Results Export**: Save and analyze matching and ranking results

## 🚀 Tech Stack

### Core Components
- **Language**: Python 3.8+
- **AI/ML**: Transformers, Sentence-Transformers
- **Retrieval**: BM25, FAISS for similarity search
- **NLP**: NLTK, spaCy for text processing
- **LLM Integration**: OpenAI GPT models for advanced processing
- **Vector Database**: FAISS for efficient similarity search

### Key Libraries
- **rank_bm25**: For BM25 retrieval
- **sentence-transformers**: For generating dense embeddings
- **transformers**: For advanced NLP models
- **torch**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key (for advanced features)
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PCTM
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🏗️ Project Structure

```
PCTM/
├── trialgpt_retrieval/      # Retrieval components
│   ├── corpus_index.py       # Corpus indexing functionality
│   ├── hybrid_fusion_retrieval.py  # Hybrid retrieval implementation
│   └── keyword_generation.py # Query keyword generation
├── trialgpt_matching/        # Matching components
│   ├── TrialGPT.py          # Main matching logic
│   └── run_matching.py      # Script to run matching process
├── trialgpt_ranking/         # Ranking components
│   ├── rank_results.py      # Main ranking implementation
│   ├── run_aggregation.py   # Script to run aggregation
│   └── trialgpt_aggregation.py  # Aggregation logic
├── dataset/                  # TREC 2022 dataset
├── results/                  # Output results
├── common/                   # Shared utilities
├── requirements.txt         # Project dependencies
└── Trailgpt.py              # Main entry point
```

## 🚦 Usage

1. Run the main script interactively:
   ```bash
   python Trailgpt.py
   ```

2. Follow the on-screen prompts to:
   - Input patient profile information
   - Select which steps to run (retrieval, matching, ranking)
   - View and save results

3. For batch processing, use the individual modules:
   ```bash
   # Run retrieval
   python -m trialgpt_retrieval.hybrid_fusion_retrieval
   
   # Run matching
   python -m trialgpt_matching.run_matching
   
   # Run ranking
   python -m trialgpt_ranking.run_aggregation
   ```

## 📊 Results

Results are saved in the `results/` directory, including:
- Retrieved trials
- Matched criteria
- Final ranked list of trials

