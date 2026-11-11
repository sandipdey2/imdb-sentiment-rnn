# IMDB Sentiment Classification with LSTM

Binary sentiment classification of IMDB movie reviews using a simple RNN/LSTM in PyTorch, with CUDA/MPS support and a controlled experimental comparison.

**GitHub Repository:** https://github.com/sandipdey2/imdb-sentiment-rnn

## Dataset
- **Name:** IMDB Movie Reviews
- **Source:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Size:** 50,000 labeled reviews (positive / negative)
- Place `imdb.csv` (columns: `review`, `sentiment`) in the same folder as the notebook.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/sandipdey2/imdb-sentiment-rnn.git
   cd imdb-sentiment-rnn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `imdb.csv` is present.
4. Open and run `deep_learning.ipynb` top to bottom.  
   Device selection automatically prefers CUDA, then Apple MPS, then CPU.

## Files
- `deep_learning.ipynb` – complete LSTM workflow and controlled experiment
- `module_summary.pdf` – analysis report with exact metrics and citations
- `requirements.txt` – environment for reproducibility
- `imdb.csv` – dataset (or documented download instructions)
