# Marvel Comics NLP Analysis

Exploring whether Marvel comic book descriptions can predict content ratings using a range of NLP techniques — from rule-based sentiment analysis to transformer embeddings and LLM classification.

## Overview

This project analyses ~11,000 Marvel comic descriptions across five content rating categories (All Ages, Teen, Teen+, Parental Advisory, Mature/Explicit) using multiple NLP approaches to understand how language correlates with content rating.

## Techniques Used

- **VADER Sentiment** — Rule-based sentiment scoring revealing a clear tonal gradient across ratings
- **BERT Sentiment** — Transformer-based contextual analysis, demonstrating the impact of domain mismatch
- **HuggingFace Embeddings** — Semantic vector representations (all-MiniLM-L6-v2) for meaning-based classification
- **Category Analysis** — Cross-referencing sentiment with Marvel imprints to validate findings
- **Random Forest Classifier** — Rating prediction using embeddings + sentiment features (54% accuracy, 5 classes)
- **LLM Zero-Shot Classification** — Claude API for training-free rating prediction (52% accuracy)

## Key Findings

- VADER outperformed BERT due to domain mismatch — simpler models can win when the domain fits better
- Semantic embeddings dominated feature importance (19 of top 20 features), showing meaning is more predictive than sentiment
- Both ML and LLM approaches excelled at rating extremes (All Ages, Mature/Explicit) but struggled with subjective middle categories
- Category analysis validated sentiment scores — imprint-level results aligned perfectly with editorial intent

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key (required for the LLM section only):
```
ANTHROPIC_API_KEY=your-key-here
```

## Requirements

See `requirements.txt` for full dependencies.

## Dataset

Marvel Comics dataset (~35,000 entries) containing comic names, descriptions, ratings, imprints, and publication metadata. Available on [Kaggle](https://www.kaggle.com/datasets/deepcontractor/marvel-comic-books).
