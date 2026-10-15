# Apple Stock Time-Series Analysis and Forecasting

A time-series portfolio project exploring historical Apple price data and a forecasting application with multiple statistical and machine-learning models.

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results and Insights](#-results-and-insights)
- [Screenshots](#-screenshots)
- [Challenges and Solutions](#-challenges-and-solutions)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)
- [Author](#-author)
- [License](#-license)

## 📖 Overview

A time-series portfolio project exploring historical Apple price data and a forecasting application with multiple statistical and machine-learning models.

## 🎯 Problem Statement

The project frames an analytical or engineering question using the data, code, and outputs available in this repository. Findings should be interpreted according to the documented dataset, methodology, and limitations.

## 🎯 Objectives

- Understand the data and project question.
- Document the implemented analytical or engineering workflow.
- Report measurable results only where supported by repository evidence.
- Make the work reproducible for reviewers.

## ✨ Features

The features below are limited to the work described in the existing repository documentation.

### Results and model validation

The historical CSV profile above was recalculated from the current repository file. The notebook could not be fully extracted within the connector's response limit in this review, and no forecast-versus-actual metric table was independently confirmed. Therefore, this README does not claim a verified forecasting accuracy, MAE, or RMSE.

A valid model comparison should document the forecast horizon, chronological split dates, baseline score, test period, model configuration, and error metric. Historical performance is not a guarantee of future price direction.

## 📂 Dataset

### Verified dataset profile

Recalculated from the repository's current `P614 DATASET.csv`:

| Metric | Result |
|---|---:|
| Rows | 2,011 |
| Columns | 7 |
| Missing cells | 0 |
| Date range | 2012-01-03 to 2019-12-30 |
| Close price minimum in the supplied file | 55.790001 |
| Close price maximum in the supplied file | 291.519989 |
| Daily volume minimum | 11,362,000 |
| Daily volume maximum | 376,530,000 |

Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`. Price units/currency and the exact adjustment methodology should be confirmed against the dataset source.

## 🛠️ Tech Stack

See the imports, dependency files, SQL scripts, and application code in the repository. Only technologies directly evidenced in those files should be listed as used.

## 🔄 Project Workflow

### Forecasting workflow

The repository documents an ensemble with multiple forecasting approaches (including ARIMA/SARIMA and exponential smoothing variants) and an XGBoost meta-learner. The Streamlit app accepts a CSV containing date and closing-price columns and supports a selectable forecast horizon of 1–60 business days.

1. Verify and sort the historical date series.
2. Inspect stationarity and temporal patterns.
3. Split data chronologically before fitting models or transformations.
4. Compare each model to a naive baseline on the same forecast horizon.
5. Evaluate holdout predictions using MAE/RMSE and other appropriate metrics.
6. Check that the app applies the same preprocessing expected by each saved model.

## 🗂️ Repository Structure

### Project files

| File | Purpose |
|---|---|
| `P614 DATASET.csv` | Historical daily price data |
| `SMA_apple.ipynb` | Analysis notebook |
| `app.py` | Streamlit forecasting application |
| `requirements.txt` | Python dependencies |
| `*.joblib` | Saved forecasting model artifacts |
| `APP_LINK.md` | App URL and usage instructions |

## ⚙️ Installation

### Prerequisites\n\nUse the runtime and dependencies required by the actual source files. Check the repository for a requirements or environment file before installing packages.\n\n### Setup\n\n\`\`\`bash\ngit clone https://github.com/sahil2003-ai/${repo}.git\ncd ${repo}\n\`\`\`

## ▶️ Usage

Open the notebook or application entry point actually present in this repository and follow its input-path and dependency instructions. Do not assume an entry script or deployment URL that is not documented.

## 📊 Results and Insights

### Results and model validation

The historical CSV profile above was recalculated from the current repository file. The notebook could not be fully extracted within the connector's response limit in this review, and no forecast-versus-actual metric table was independently confirmed. Therefore, this README does not claim a verified forecasting accuracy, MAE, or RMSE.

A valid model comparison should document the forecast horizon, chronological split dates, baseline score, test period, model configuration, and error metric. Historical performance is not a guarantee of future price direction.

## Additional Project Documentation\n\n### Run the app

1. Create a Python environment and install `requirements.txt`.
2. Launch the app from the repository root:

   ```bash
   streamlit run app.py
   ```

3. Upload a CSV with date and closing-price fields and select the correct columns.
4. Select a forecast horizon and generate predictions.

Live app link, as documented in `APP_LINK.md`: [Stock Price Forecast App](https://stock-market-analysis---apple-qj4jebpk8mthmdmysuy5vy.streamlit.app/).

### Author

Sahil Gaikwad · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)\n## 🖼️ Screenshots

No verified screenshot path was identified in the existing README. Add real screenshots from the repository only when the corresponding image files exist.

## 🧩 Challenges and Solutions

### Validation and reproducibility

| Challenge | Approach |
|---|---|
| Metrics can be mistaken for independently verified results | Preserve the evidence status stated in the existing documentation. |
| Data sources, filters, or execution environments may differ | Confirm source paths, schema, dependencies, and calculation definitions when rerunning. |
| Screenshots and dashboard outputs may not be independently accessible | Only link to real files present in the repository. |

## ⚠️ Limitations

### Limitations

- The dataset spans historical observations only; forecasts are uncertain.
- Price adjustment, source, and licensing details need explicit documentation.
- Saved forecasts need chronological holdout evaluation and a naive baseline before performance claims are made.
- This is an educational modelling project, not investment advice.

## 🚀 Future Improvements

- [ ] Document dataset source, license, and data dictionary.\n- [ ] Add reproducible execution instructions and dependency versions.\n- [ ] Export validated result tables and genuine screenshots.\n- [ ] Add tests or cross-tool checks for important calculations.

## 📚 What I Learned

### Skills demonstrated

Python, pandas, time-series preparation, statistical forecasting, model comparison, validation design, model serialization, and Streamlit.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

No verified license statement was found in the README. Check the repository's LICENSE file and the source dataset's reuse terms before redistribution.
