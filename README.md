# ❤️ Heart Disease Prediction

![Flask](https://img.shields.io/badge/Flask-3.0+-black?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-blue?style=for-the-badge&logo=scikit-learn)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)

Heart Disease Prediction is a machine learning-powered web application designed to assess an individual's risk of heart disease based on clinical metrics. Built for accuracy and ease of use, it provides real-time predictions and curated health advice.

## ✨ Key Features

*   **🧠 Machine Learning:** Utilizes a tuned Logistic Regression model with Recursive Feature Elimination (RFE) for accurate predictions.
*   **🔒 Secure Access:** Implements user authentication (register/login) using Flask-Login and Werkzeug password hashing.
*   **⚡ Real-Time Inference:** Instant risk assessment (High/Low) and probability scoring based on user-inputted medical data.
*   **🩺 Curated Advice:** Dynamically generates tailored medical recommendations and symptom warnings.
*   **🚀 Production Ready:** Configured with Waitress for reliable WSGI production deployment.

## 🛠️ Tech Stack

**Frontend:**
*   HTML5 & CSS3
*   Jinja2 Templating

**Backend & ML:**
*   Flask & Flask-SQLAlchemy
*   scikit-learn, Pandas, NumPy
*   SQLite3

## 🚀 Getting Started

### Prerequisites
*   [Python](https://www.python.org/downloads/) 3.x installed
*   pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/heart_disease_prediction.git
    cd heart_disease_prediction
    ```

2.  **Set up environment and install dependencies:**
    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Train the Model (Generates Artifacts):**
    ```bash
    python train_model.py
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Open [http://localhost:5000](http://localhost:5000) in your browser.

## 📂 Project Structure

```text
heart_disease_prediction/
├── static/            # CSS stylesheets and images
├── templates/         # HTML Jinja2 templates
├── app.py             # Flask application & routing
├── train_model.py     # ML pipeline & model generation
├── requirements.txt   # Project dependencies
└── users.db           # SQLite database (auto-generated)
