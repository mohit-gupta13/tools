# Student Exam Performance Prediction

## Project Overview
This project is an end-to-end Machine Learning web application designed to predict a student's **Math Score** based on various factors such as gender, ethnicity, parental education level, lunch type, and test preparation scores.

It features a modular pipeline for data ingestion, transformation, model training, and a Flask-based web interface for real-time predictions.

## 🚀 Features
- **Data Ingestion**: Automates the process of loading and splitting data.
- **Data Transformation**: Handles categorical encoding and numerical scaling with custom pipelines.
- **Model Training**: Evaluates multiple regression models (Linear Regression, Random Forest, XGBoost, CatBoost, etc.) and selects the best-performing one.
- **Web Interface**: User-friendly form to input student details and get instant score predictions.

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Framework**: Flask
- **ML Libraries**: Scikit-Learn (v1.5.2), CatBoost, XGBoost, Pandas, NumPy
- **Serialization**: Dill, Pickle

## 📁 Project Structure
```text
├── artifacts/              # Saved models, preprocessors, and CSV splits
├── notebook/               # Jupyter notebooks and raw dataset
│   └── data/
│       └── StudentsPerformance.csv
├── src/                    # Source code
│   ├── components/         # Core ML steps: Ingestion, Transformation, Training
│   ├── pipeline/           # Prediction and training pipelines
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging configuration
│   └── utils.py            # Utility functions for saving/loading objects
├── templates/              # HTML frontend files
├── app.py                  # Flask application entry point
├── requirements.txt        # Project dependencies
└── setup.py                # Package configuration
```

## ⚙️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd tools
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Running the Project

### 1. Training the Model
To run the complete data pipeline and train the model, execute:
```bash
python src/components/data_ingestion.py
```
This will generate `model.pkl` and `preprocessor.pkl` in the `artifacts/` directory.

### 2. Launching the Web App
To start the Flask server locally:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/predict`.

## 📌 Recent Fixes and Improvements
- Fixed routing mismatch for the prediction endpoint.
- Resolved scikit-learn version compatibility issues with CatBoost (pinned to 1.5.2).
- Improved form validation and label accuracy in the user interface.
- Standardized data column names (`race/ethnicity`) across the pipeline.