# Anime Recommender System (Beta)

*By Juvénis KABORE*

---

## Overview

**Anime Recommender System** helps users discover new anime tailored to their interests using a combination of content-based and collaborative filtering. Built as a modern web app, it offers an intuitive user interface and employs robust machine learning to generate relevant recommendations.

**Status:** _Beta release. Core functionality is complete, but additional features and refinements are planned for future versions._

---

## Purpose & Audience

This system addresses the challenge anime fans face in finding what to watch next by leveraging data-driven recommendations. 
It is intended for:
- Anime streaming services seeking enhanced discovery tools,
- Communities or sites wanting to engage users,
- Technical evaluators exploring best practices in recommendation engines and MLOps.

---

## Key Features

- **Smart Anime Search:** Tolerant to typos and partial input.
- **Content-based Recommendations:** Suggests similar anime by metadata (genres, tags, etc.).
- **Hybrid Recommendations:** Combines collaborative filtering (user history) and content similarity for personalized results.
- **Modern, Responsive UI:** Built with Streamlit for instant feedback and a visually appealing layout.
- **Cloud-native Backend:** Uses Supabase for scalable data storage.
- **Customizable:** Adjust the blend of personalization and similarity to suit user preferences.
- **Popular Picks:** Highlights trending anime if no specific search is made.

---

## Tech Stack

- **Frontend:** Streamlit (Python)
- **Backend/Database:** Supabase (managed PostgreSQL)
- **Recommendation Engine:** 
  - scikit-surprise (SVD collaborative filtering)
  - Numpy-based embeddings (content similarity)
- **Fuzzy Search:** rapidfuzz
- **Data Analysis:** pandas, numpy
- **Serialization:** pickle

---

## Project Structure

```
Anime_Recommender_System/
├── app.py                   # Main Streamlit app
├── recommender-system.ipynb # Jupyter notebook (modeling & data engineering)
├── svd_model.pkl            # Trained collaborative filtering model
├── anime_embeddings.npy     # Encoded anime features
└── .streamlit/
    └── secrets.toml         # Supabase credentials
```

---

## Quick Start

### Requirements

- Python 3.8+
- Streamlit, pandas, numpy, rapidfuzz, scikit-surprise, supabase-py
- Supabase account and API keys
- Model files: `svd_model.pkl`, `anime_embeddings.npy`

### Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Spykabore15/Anime_Recommender_System.git
   cd Anime_Recommender_System
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure secrets:**  
   Create `.streamlit/secrets.toml`:
   ```toml
   SUPABASE_URL = "your-supabase-instance-url"
   SUPABASE_KEY = "your-supabase-API-key"
   ```

4. **Ensure model files are present:**  
   Place `svd_model.pkl` and `anime_embeddings.npy` in the repo root.

5. **Launch the app:**
   ```sh
   streamlit run app.py
   ```

---

## Usage

- **Search** for your favorite anime. Fuzzy matching retrieves correct titles even with typos.
- As a **new user**: Click "Find Similar Anime" for recommendations based on your selection's attributes.
- As an **existing user**: Enter your User ID to receive hybrid recommendations tuned to your viewing and rating history. Adjust personalization as needed.

---

## Configuration & Deployment Notes

- **Supabase credentials** must be present in `.streamlit/secrets.toml`.
- **User IDs** for testing the hybrid recommendations are currently valid from 1 to 10,000 and can be adjusted as the dataset grows.
- **Model files** are required and must be generated with the included Jupyter notebook or provided by the author for production use.

---

## Example

_Search for "Attack on Titan" → receive a ranked list of similar titles with covers, genres, and descriptions. Existing user IDs return even more personalized lists._


---

## Roadmap / Planned Improvements

- **User Authentication:** Persistent profiles, ratings, login.
- **Interactive Feedback:** Live rating to retrain/update recommendations in real-time.
- **API/Frontend Separation:** Containerized deployment for scalability.
- **Advanced Recommendation Methods:** Integration of deep learning or graph-based techniques.
- **Mobile UX Enhancements**
- **Expanded Metadata:** Streaming links, trailers, and community reviews.
- **Open Source License:** Will be specified in a future release.

---

## License

_No license is currently specified. Please contact the author, Juvénis KABORE, for usage or contribution terms._

---

_For more details, technical inquiries, or collaboration requests, please reach out to the author: **Juvénis KABORE**._

---
