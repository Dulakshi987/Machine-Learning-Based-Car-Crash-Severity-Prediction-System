# Machine-Learning-Based-Car-Crash-Severity-Prediction-System

This project builds an end-to-end machine learning system to predict **car crash severity** (Minor Injury, Severe Injury, Fatal) using tabular data from the Kaggle competition **"Car Crash Severity Prediction"**.

### Project Highlights
- Full Exploratory Data Analysis (EDA): Missing values, outliers, skewness, correlation heatmap, class imbalance visualization
- Data Preprocessing: Median/mode imputation, label encoding, IQR outlier handling, SMOTE balancing, scaling
- Model Training & Comparison:
  - Random Forest (ensemble baseline)
  - XGBoost (best model – weighted F1-score: 0.799, accuracy: 0.801)
  - MLP Classifier (neural network/deep learning baseline)
- Best model selected based on weighted F1-score → deployed as Flask web app
- Real-time prediction UI with Bootstrap + popup results
- Kaggle submission: Public leaderboard score ≈ 0.575

### Tech Stack
- **Core**: Python 3, Pandas, NumPy
- **ML**: XGBoost, imbalanced-learn (SMOTE), joblib, Random Forest Classifier
- **Deep Learning**: MLP Classifier
- **Visualization**: Matplotlib, Seaborn
- **Web Deployment**: Flask (backend), HTML/CSS/Bootstrap/JavaScript (frontend)
- **Storage**: Pickle (models), JSON (metadata)

### Project Structure

```text
1.CI_CIS6005-II/
├── data/                        
│   ├── car_crash_train.csv      
│   ├── car_crash_test.csv       
│   └── sample_submission.csv    
├── models/                      
│   ├── crash_severity_system.pkl    
│   └── preprocessing_artifacts.pkl 
├── output/                     
│   ├── *.png                    
│   └── eda_insights_report.txt  
├── metadata/                    
│   └── dataset_metadata.json    
├── notebooks/                  
│   └── CI_CIS6005_WRIT.ipynb    
├── app/                         
│   ├── app.py                  
│   ├── templates/
│   │   └── index.html           
│   └── static/
│       ├── css/                 
│       └── js/                  
└── README.md
```

### Setup & Run Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/car-crash-severity-prediction.git
   cd car-crash-severity-prediction/1.CI_CIS6005-II

2. **Run Flask Web App**
 ```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3. **Create Virtual Environment**
 ```bash
cd app
python app.py
