"""
Hospital Patient Readmission Prediction - Data Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HospitalReadmissionPreprocessor:
    """
    Comprehensive preprocessing pipeline for hospital readmission prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_names = []
        
    def load_sample_data(self):
        """
        Generate sample hospital data for demonstration
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic patient data
        data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.normal(65, 15, n_samples).clip(18, 95),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples, p=[0.6, 0.2, 0.2]),
            'discharge_disposition': np.random.choice(['Home', 'SNF', 'Home_Health', 'AMA'], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'admission_source': np.random.choice(['Emergency_Room', 'Physician_Referral', 'Transfer'], n_samples, p=[0.5, 0.3, 0.2]),
            'length_of_stay': np.random.exponential(4, n_samples).clip(1, 30),
            'num_procedures': np.random.poisson(2, n_samples),
            'num_medications': np.random.poisson(8, n_samples),
            'num_diagnoses': np.random.poisson(5, n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'heart_disease': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'previous_admissions': np.random.poisson(1, n_samples),
            'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self_Pay'], n_samples, p=[0.4, 0.2, 0.3, 0.1]),
            'admission_date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
            'discharge_date': None,  # Will calculate based on length_of_stay
            'lab_glucose': np.random.normal(120, 30, n_samples).clip(60, 300),
            'lab_hemoglobin': np.random.normal(12, 2, n_samples).clip(8, 18),
            'readmitted_30_days': None  # Target variable - will calculate
        }
        
        df = pd.DataFrame(data)
        
        # Calculate discharge dates
        df['discharge_date'] = df.apply(lambda row: 
            row['admission_date'] + timedelta(days=int(row['length_of_stay'])), axis=1)
        
        # Introduce some missing values to simulate real data
        missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_indices[:50], 'lab_glucose'] = np.nan
        df.loc[missing_indices[50:], 'lab_hemoglobin'] = np.nan
        
        # Generate target variable based on risk factors
        risk_score = (
            (df['age'] > 70).astype(int) * 0.3 +
            (df['previous_admissions'] > 2).astype(int) * 0.4 +
            (df['length_of_stay'] > 7).astype(int) * 0.2 +
            df['diabetes'] * 0.2 +
            df['heart_disease'] * 0.2 +
            (df['discharge_disposition'] == 'AMA').astype(int) * 0.5 +
            np.random.normal(0, 0.2, len(df))
        )
        
        df['readmitted_30_days'] = (risk_score > 0.7).astype(int)
        
        return df
    
    def data_quality_assessment(self, df):
        """
        Assess data quality and identify issues
        """
        print("=== DATA QUALITY ASSESSMENT ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nMissing values per column:")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            for col, count in missing_data.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
        else:
            print("  No missing values detected")
        
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        print(f"\nTarget variable distribution:")
        print(df['readmitted_30_days'].value_counts(normalize=True))
        
        return missing_data
    
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate imputation strategies
        """
        print("\n=== HANDLING MISSING VALUES ===")
        df_processed = df.copy()
        
        # Numerical columns - use KNN imputation for lab values
        numerical_cols_missing = ['lab_glucose', 'lab_hemoglobin']
        numerical_cols_missing = [col for col in numerical_cols_missing if col in df.columns and df[col].isnull().any()]
        
        if numerical_cols_missing:
            print(f"Applying KNN imputation to: {numerical_cols_missing}")
            knn_imputer = KNNImputer(n_neighbors=5)
            df_processed[numerical_cols_missing] = knn_imputer.fit_transform(df_processed[numerical_cols_missing])
        
        # Categorical columns - use mode imputation
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['admission_date', 'discharge_date']]
        
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                mode_value = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_value, inplace=True)
                print(f"Filled missing {col} with mode: {mode_value}")
        
        return df_processed
    
    def feature_engineering(self, df):
        """
        Create new features from existing data
        """
        print("\n=== FEATURE ENGINEERING ===")
        df_features = df.copy()
        
        # 1. Charlson Comorbidity Index (simplified)
        print("Creating Charlson Comorbidity Index...")
        df_features['charlson_score'] = (
            df_features['diabetes'] * 1 +
            df_features['heart_disease'] * 1 +
            df_features['hypertension'] * 1
        )
        
        # 2. Age groups
        print("Creating age groups...")
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 30, 50, 65, 80, 100], 
                                        labels=['Young', 'Middle_Age', 'Senior', 'Elderly', 'Very_Elderly'])
        
        # 3. Length of stay categories
        print("Creating length of stay categories...")
        df_features['los_category'] = pd.cut(df_features['length_of_stay'], 
                                           bins=[0, 3, 7, 14, 30], 
                                           labels=['Short', 'Medium', 'Long', 'Very_Long'])
        
        # 4. High-risk indicators
        print("Creating high-risk indicators...")
        df_features['high_risk_age'] = (df_features['age'] >= 75).astype(int)
        df_features['frequent_admitter'] = (df_features['previous_admissions'] >= 3).astype(int)
        df_features['polypharmacy'] = (df_features['num_medications'] >= 10).astype(int)
        df_features['multiple_procedures'] = (df_features['num_procedures'] >= 3).astype(int)
        
        # 5. Medication complexity score
        print("Creating medication complexity score...")
        df_features['med_complexity_score'] = (
            df_features['num_medications'] * 0.3 +
            df_features['diabetes'] * 2 +  # Diabetes medications are complex
            df_features['heart_disease'] * 1.5
        )
        
        # 6. Lab value categories
        print("Creating lab value categories...")
        df_features['glucose_category'] = pd.cut(df_features['lab_glucose'], 
                                               bins=[0, 100, 140, 200, 400], 
                                               labels=['Normal', 'Prediabetic', 'Diabetic', 'Severe'])
        
        df_features['hemoglobin_category'] = pd.cut(df_features['lab_hemoglobin'], 
                                                  bins=[0, 10, 12, 16, 20], 
                                                  labels=['Severe_Anemia', 'Mild_Anemia', 'Normal', 'High'])
        
        # 7. Seasonal patterns
        print("Creating seasonal features...")
        df_features['admission_month'] = df_features['admission_date'].dt.month
        df_features['admission_season'] = df_features['admission_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # 8. Day of week for admission (weekends might be different)
        print("Creating temporal features...")
        df_features['admission_day_of_week'] = df_features['admission_date'].dt.dayofweek
        df_features['weekend_admission'] = (df_features['admission_day_of_week'].isin([5, 6])).astype(int)
        
        print(f"Feature engineering complete. Added {len(df_features.columns) - len(df.columns)} new features.")
        
        return df_features
    
    def prepare_for_modeling(self, df):
        """
        Prepare data for machine learning models
        """
        print("\n=== PREPARING FOR MODELING ===")
        
        # Separate features and target
        target = 'readmitted_30_days'
        
        # Columns to exclude from features
        exclude_cols = ['patient_id', 'admission_date', 'discharge_date', target]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Identify categorical and numerical columns
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.column_transformer = ColumnTransformer([
            ('numerical', numerical_pipeline, numerical_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        # Fit and transform the data
        X_processed = self.column_transformer.fit_transform(X)
        
        # Get feature names for the processed data
        numerical_feature_names = numerical_features
        categorical_feature_names = []
        
        # Get feature names from one-hot encoder
        onehot_encoder = self.column_transformer.named_transformers_['categorical']['onehot']
        if hasattr(onehot_encoder, 'get_feature_names_out'):
            categorical_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
        else:
            # Fallback for older sklearn versions
            categorical_feature_names = [f"{cat}_{val}" for cat in categorical_features 
                                       for val in onehot_encoder.categories_[categorical_features.index(cat)][1:]]
        
        self.feature_names = numerical_feature_names + categorical_feature_names
        
        print(f"Final feature matrix shape: {X_processed.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        return X_processed, y, self.feature_names
    
    def run_complete_pipeline(self):
        """
        Execute the complete preprocessing pipeline
        """
        print("HOSPITAL READMISSION PREDICTION - PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        print("Step 1: Loading sample data...")
        df = self.load_sample_data()
        
        # Step 2: Data quality assessment
        missing_data = self.data_quality_assessment(df)
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 4: Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Step 5: Prepare for modeling
        X_processed, y, feature_names = self.prepare_for_modeling(df_features)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Ready for model training with {X_processed.shape[1]} features")
        print("=" * 60)
        
        return X_processed, y, feature_names, df_features

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = HospitalReadmissionPreprocessor()
    
    # Run complete pipeline
    X_processed, y, feature_names, df_original = preprocessor.run_complete_pipeline()
    
    # Display sample of processed data
    print(f"\nSample of processed features (first 5 features, first 10 rows):")
    print(X_processed[:10, :5])
    
    print(f"\nTarget variable distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} ({count/len(y)*100:.1f}%)")
