# AI Fake News Detection

AI powered web application for fake news detection. This application analyzes the writing style and linguistic patterns of news articles to distinguish between legitimate journalism and fabricated content.

## Features
- **Analysis:** Input a news title and text to get a verdict (REAL or FAKE)
- **Confidence Score:** Displays the model's certainty percentage
- **Modern UI:** Responsive Frontend built with React
- **API:** Fast and efficient Backend built with FastAPI

## Prerequisites

* **Python** (for the backend)
* **Node.js & npm** (for the frontend)
* **NVIDIA CUDA Toolkit** (Optional - Recommended for GPU acceleration)

---

## Installation

### 1. Clone the repository to your local machine:
```bash
git clone https://github.com/Andrei25i/Fake-News-Detection.git
cd fake-news-detector
```
### 2. Create a virtual environment, activate it, and install dependencies:
```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```
### 3. Download Models (required)

Download the trained model files from [HuggingFace](https://huggingface.co/Andrei25i/fake-news-detection/tree/main).
Extract and place the `models` directory in the root of the project.

### 4. Configuration (optional)
Create an `.env` file in the root of the `frontend` and the `backend` folders and configure the backend url.
```bash
# backend/.env
HOST=
PORT=
```

```bash
# frontend/.env
VITE_API_URL=
```
### 5. Start the backend
```bash
cd backend
python main.py
```

### 6. Setup and Run Frontend:

Open a new terminal window and run:
```bash
cd frontend
npm install
npm run dev
```

## Training (optional)

If you wish to retrain the models from scratch or experiment with new datasets, follow the workflow below using the Jupyter Notebooks located in the `notebooks` directory.

### 1. Download the datasets
Download these datasets and put them in a folder named `data` in the root of the project.

[Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

[BBC News](https://www.kaggle.com/datasets/gpreda/bbc-news)

### 2. Data Preparation
Run **`Merging_Datasets.ipynb`**
* **Purpose:** Merges the **WELFake** dataset with the **BBC News** dataset to create a balanced dataset.
* **Output:** A cleaned and combined CSV file ready for training.

### 3. Logistic Regression (Classic Model)
Run **`Logistic_Regression.ipynb`**
* **Purpose:** Trains the baseline statistical model using TF-IDF and Logistic Regression.
* **Output:** Generates the `classic_model` folder containing `model_lr.pkl` and `tfidf_vectorizer.pkl`.

### 4. BERT Fine-Tuning (Deep Learning)
Run **`Deep_Learning_Bert.ipynb`**
* **Purpose:** Fine-tunes the `bert-base-cased` model on the hybrid dataset.
* **System Requirement:** An Nvidia GPU is highly recommended.
* **Output:** Generates the `bert_model` directory containing the model weights and configuration.