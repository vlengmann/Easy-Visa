
# Easy Visa

**Machine learning project solution** that profiles visa approvals of US applicants via a use of a **classification model**, visualizes using **Dash** and deployed using **Azure**


Built with Python, XGBoost, and Plotly Dash, this interactive dashboard helps users understand the likelihood of visa approvals based on given characteristics and employee qualifications.

**Try the app here:**[https://easy-visa-rg.azurewebsites.net](https://easy-visa-rg.azurewebsites.net)

---
## Project Timeline
* **Original Completion:** June 2024
* **Dash App Integration:** December 2025
* **Azure Deployment:** January 2026
* **Github Upload:** January 2026
This project was completed June 2024 and re-exported for documentation, deployment and presentation.

## Data  


* `case_id`: ID of each visa application
* `continent`: Information of continent the employee
* `education_of_employee`: Information of education of the employee
* `has_job_experience`: Does the employee has any job experience? Y= Yes; N = No
* `requires_job_training`: Does the employee require any job training? Y = Yes; N = No
* `no_of_employees`: Number of employees in the employer's company
* `yr_of_estab`: Year in which the employer's company was established
* `region_of_employment`: Information of foreign worker's intended region of employment in the US.
* `prevailing_wage`:  Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment.
* `unit_of_wage`: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
* `full_time_position`: Is the position of work full-time? Y = Full Time Position; N = Part Time Position
* `case_status`:  Flag indicating if the Visa was certified or denied
## Project Overview

This project analyzes US visa application data to predict certification outcomes using machine learning. The solution includes:
- Exploratory data analysis and feature engineering
- Multiple classification models (Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost)
- Production-ready web application with interactive visualizations
- Cloud deployment on Azure App Service



## Project Structure

```
Easy-Visa/
├── app.py                      # Main Dash application
├── requirements.txt            # Python dependencies
├── startup.txt                 # Azure startup command
├── AZURE_DEPLOYMENT.md         # Deployment guide
├── model/
│   ├── train.py               # Model training script
│   ├── preprocess.py          # Data preprocessing functions
│   ├── model.pkl              # Trained XGBoost model
│   ├── feature_columns.pkl    # Feature names
│   ├── feature_importances.npy # Feature importance values
│   ├── metrics.json           # Model performance metrics
│   └── confusion_matrix.npy   # Confusion matrix
├── utils/
│   ├── data_ingestion.py      # Data loading utilities
│   ├── load_model.py          # Model loading functions
│   └── predict.py             # Prediction utilities
├── data/
│   └── visa_data.csv          # Training dataset
├── notebooks/
│   └── EasyVisa full code.ipynb # Exploratory data analysis
└── .github/
    └── workflows/
        └── main_easy-visa-rg.yml # GitHub Actions deployment workflow
```
## Methods & Analysis 
visualize confusuion matrix 
Accuracy, recall, precsion, F1
Hyperparameter tuned models
Decision Tree, Bagging (grid search cv), bagging estimator
random forest (grid search cv)

Gradient Boost(ADA Boost, decion tree grid search cv)
Feature importance found using normalizedGini imprtance

ADA (grid search cv)
XGB (grid search cv)

Model Performance Comparison


## App Features

- **Interactive Web Interface** - User-friendly form for visa prediction
- **Real-time Predictions** - Instant certification likelihood with confidence scores
- **XGBoost ML Model** - 74.5% accuracy, 82.2% F1 score
- **Feature Importance Visualization** - Interactive Plotly chart showing key decision factors
- **Responsive Design** - Bootstrap-styled UI that works on all devices
- **Cloud Hosted** - Azure App Service with automated CI/CD
- **Automated Deployment** - GitHub Actions pipeline for continuous delivery
## Model Visualization and Deployment

- Clear prediction results with displayed accuracy
- Professional interface with input capabilities
- Feature importance mapping
- Deployment through Azure for reproducability
## Installation and Setup


### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/vlengmann/Easy-Visa.git
   cd Easy-Visa
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open browser**
   Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050)
## Model Training 


To retrain the model with new data:

1. Place your CSV data in `Easy-Visa/data/visa_data.csv`
2. Run the training script:
   ```bash
   python -m model.train
   ```
3. New model artifacts will be saved in the `model/` directory:
   - `model.pkl` - Trained model
   - `feature_columns.pkl` - Feature names
   - `feature_importances.npy` - Feature importance scores
   - `metrics.json` - Performance metrics
   - `confusion_matrix.npy` - Confusion matrix

---

## Cloud Deployment

The application is deployed on **Azure App Service** with automated GitHub Actions CI/CD pipeline.

### Deploy Your Own Instance

See [AZURE_DEPLOYMENT.md](Easy-Visa/AZURE_DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy:**
1. Fork this repository
2. Create Azure App Service (Python 3.11)
3. Connect GitHub repository in Deployment Center
4. Select branch: `main`
5. Azure will automatically build and deploy  


**Automatic Redeployment:**
Every push to the `main` branch triggers an automatic deployment via GitHub Actions.
## License

[MIT](https://choosealicense.com/licenses/mit/)

