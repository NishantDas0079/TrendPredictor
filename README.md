# 📈 TrendPredictor

[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)](https://trendpredictor.onrender.com)



**TrendPredictor** is an interactive time series forecasting web application built with [Streamlit](https://streamlit.io/) and [Facebook Prophet](https://facebook.github.io/prophet/). It allows users to explore multiple time series, adjust hyperparameters, train models, and detect anomalies—all through a beautiful, modern UI.

---

## ✨ Features

- 📊 **Multi‑series support** – Choose from over 200 different time series.
- ⚙️ **Hyperparameter tuning** – Adjust Prophet’s `changepoint_prior_scale`, `seasonality_prior_scale`, and seasonality modes.
- 💾 **Model persistence** – Save trained models and reuse them later.
- 🚨 **Anomaly detection** – Automatically flag points where actual values fall outside forecast confidence intervals.
- 🎨 **Modern UI** – Custom CSS with animated header, metric cards, and smooth transitions.
- 📥 **Download forecasts** – Export predictions as CSV.

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|-----------------|------------------------------------|
| Frontend        | [Streamlit](https://streamlit.io/) |
| Forecasting     | [Prophet](https://facebook.github.io/prophet/) |
| Visualization   | [Plotly](https://plotly.com/)      |
| Data Handling   | Pandas, NumPy                      |
| Deployment      | [Render](https://render.com/)      |

---

## 🚀 Live Demo

🔗 **[Try TrendPredictor here]()**  
*(Replace with your actual Render URL once deployed.)*

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Steps

# 1. Clone the Repository
 ```bash
 git clone https://github.com/NishantDas0079/TrendPredictor.git
 cd TrendPredictor
 ```

# 2. Create and Activate virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

# 3. Install dependencies
```
pip install -r requirements.txt
```

# 4. Downloading the Dataset

The dataset `(y_amazon-google-large.csv)` is not included in the repository due to its size `(~115 MB)`.

Download it from the `UCI Machine Learning Repository` and place it in `data/raw/`.


# 5. Run the App
```
streamlit run src/app.py
```

# 🧪 Usage

Select a series from the sidebar dropdown.

Adjust the forecast horizon and test size.

Tune hyperparameters (optional).

Click `"Train & Forecast"`.

View evaluation metrics, plots, and any detected anomalies.

Download the future forecast as a `CSV`

# 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the `issues page`.
