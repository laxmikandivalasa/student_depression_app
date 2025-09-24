
# Student Depression Prediction App 🧠

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **Streamlit web app** to predict student depression using a **Random Forest Classifier**. Input student details interactively and get real-time predictions with probability and model metrics.

---

## Features
- View dataset and basic stats
- Model evaluation: accuracy, confusion matrix, ROC curve
- Interactive prediction for new student inputs

---

## Installation
```bash
git clone <repo_url>
cd student_depression_app
python -m venv venv
# activate environment
pip install -r requirements.txt
streamlit run app.py
````

---

## Dataset

* CSV file: `depression_student_dataset.csv`
* Features: demographics, academic performance, lifestyle, mental health
* Target: `Depression` (`Yes`/`No`)

---

## Usage

* Input student data via sliders and dropdowns
* See prediction (`Depression` / `No Depression`) and probability
* View model metrics and visualizations

---

## License

This project is licensed under the MIT License.
