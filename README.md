# INSE-6450

Version control for our group project.

## Project overview
This repository contains a Google Colab/Jupyter notebook for exploratory data analysis on the Criteo uplift dataset. The notebook loads the dataset directly from Hugging Face and performs EDA to inspect data distributions, identify missing values, summarize feature behavior, and evaluate uplift-related patterns that support downstream modeling and analysis.

## Google Colab Notebook
- https://colab.research.google.com/drive/1gKct2Ir_g3hEOOPjexrCBOU7skpkdSPB?usp=sharing

## Output Artifact
The project output artifact is published on GitHub Releases:
- https://github.com/saopa28/INSE-6450/releases/tag/v1.0-engineered

## Dependencies
Install the required Python packages before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

The notebook also uses the Hugging Face dataset loader to access:

```python
hf://datasets/criteo/criteo-uplift/criteo-research-uplift-v2.1.csv.gz
```

## How to run
Option 1: Run in Google Colab
1. Open the Colab link above.
2. Upload or clone this repository if needed.
3. Run the cells in `INSE6450.ipynb` in order.

Option 2: Run locally
```bash
git clone https://github.com/saopa28/INSE-6450.git
cd INSE-6450
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook INSE6450.ipynb
```

Then open `INSE6450.ipynb` in Jupyter and execute the cells.

## Notes
- The dataset used by the notebook is large (approximately 14 million rows, 16 columns).
- The notebook is designed primarily for exploratory analysis and visualization in a notebook environment.
- No dedicated output directory is committed to the repository at this time.
