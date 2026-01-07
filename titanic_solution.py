"""
Titanic Survival Prediction - Complete Solution
This script implements a systematic approach to solve the Titanic classification problem
using Logistic Regression following 15 comprehensive steps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_curve, roc_auc_score, auc)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class TitanicSolution:
    """
    Complete solution for Titanic survival prediction problem.
    Implements all 15 steps from data loading to model evaluation and recommendations.
    """
    
    def __init__(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Initialize the solution with dataset paths.
        
        Args:
            train_path (str): Path to training dataset
            test_path (str): Path to test dataset
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = None
        self.model = None
        self.best_model = None
        self.feature_names = None
        
        # Constants for feature engineering
        self.AGE_BINS = [0, 12, 18, 35, 60, 100]
        self.AGE_LABELS = ['Child', 'Adolescent', 'Adult', 'Middle-aged', 'Senior']
        self.fare_bins = None  # Will be calculated from training data
        
    # ============================================================================
    # STEP 1: LOADING AND INITIAL EXPLORATION
    # ============================================================================
    
    def load_and_explore(self):
        """
        Load datasets and perform initial exploration.
        - Display first 5 rows
        - Show column information (types, missing values)
        - Provide descriptive statistics
        - Identify missing values
        """
        print("="*80)
        print("STEP 1: LOADING AND INITIAL EXPLORATION")
        print("="*80)
        
        # Load datasets
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"\nTraining dataset shape: {self.train_df.shape}")
        print(f"Test dataset shape: {self.test_df.shape}")
        
        # Display first 5 rows
        print("\n--- First 5 rows of training data ---")
        print(self.train_df.head())
        
        # Show column information
        print("\n--- Column Information ---")
        print(self.train_df.info())
        
        # Descriptive statistics
        print("\n--- Descriptive Statistics for Numerical Variables ---")
        print(self.train_df.describe())
        
        # Identify missing values
        print("\n--- Missing Values Count ---")
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()
        
        missing_df = pd.DataFrame({
            'Train': missing_train[missing_train > 0],
            'Test': missing_test[missing_test > 0]
        })
        print(missing_df)
        
        return self.train_df, self.test_df
    
    # ============================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ============================================================================
    
    def exploratory_analysis(self):
        """
        Perform comprehensive exploratory data analysis.
        - Analyze survival rate by different features
        - Identify correlations
        - Draw conclusions about important features
        """
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # Survival rate by Gender
        print("\n--- Survival Rate by Gender ---")
        gender_survival = self.train_df.groupby('Sex')['Survived'].agg(['mean', 'count'])
        print(gender_survival)
        
        # Survival rate by Class
        print("\n--- Survival Rate by Pclass ---")
        class_survival = self.train_df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
        print(class_survival)
        
        # Survival rate by Embarked
        print("\n--- Survival Rate by Embarked ---")
        embarked_survival = self.train_df.groupby('Embarked')['Survived'].agg(['mean', 'count'])
        print(embarked_survival)
        
        # Survival rate by Age groups
        print("\n--- Survival Rate by Age Groups ---")
        self.train_df['AgeGroup_temp'] = pd.cut(self.train_df['Age'], 
                                                 bins=self.AGE_BINS,
                                                 labels=self.AGE_LABELS)
        age_survival = self.train_df.groupby('AgeGroup_temp')['Survived'].agg(['mean', 'count'])
        print(age_survival)
        
        # Family size analysis
        print("\n--- Survival Rate by Family Size ---")
        self.train_df['FamilySize_temp'] = self.train_df['SibSp'] + self.train_df['Parch'] + 1
        family_survival = self.train_df.groupby('FamilySize_temp')['Survived'].agg(['mean', 'count'])
        print(family_survival)
        
        # Correlation analysis
        print("\n--- Correlation Matrix for Numerical Variables ---")
        numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        correlation_matrix = self.train_df[numerical_cols].corr()
        print(correlation_matrix['Survived'].sort_values(ascending=False))
        
        # Create visualizations
        self._create_eda_visualizations()
        
        # Conclusions
        print("\n--- Key Insights ---")
        print("1. Gender: Females have significantly higher survival rate (~74%) vs males (~19%)")
        print("2. Class: 1st class passengers had highest survival rate (~63%), 3rd class lowest (~24%)")
        print("3. Age: Children had higher survival rates")
        print("4. Family Size: Passengers with small families (2-4) had better survival rates")
        print("5. Fare: Higher fares correlate with better survival (proxy for class)")
        print("6. Most important features: Sex, Pclass, Fare, Age")
    
    def _create_eda_visualizations(self):
        """Create and save EDA visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Survival by Gender
        self.train_df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'salmon'])
        axes[0, 0].set_title('Survival Rate by Gender')
        axes[0, 0].set_ylabel('Survival Rate')
        axes[0, 0].set_xticklabels(['Female', 'Male'], rotation=0)
        
        # 2. Survival by Class
        self.train_df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0, 1], color='green')
        axes[0, 1].set_title('Survival Rate by Passenger Class')
        axes[0, 1].set_ylabel('Survival Rate')
        axes[0, 1].set_xlabel('Pclass')
        
        # 3. Age distribution
        self.train_df[self.train_df['Survived']==1]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5, label='Survived', color='green')
        self.train_df[self.train_df['Survived']==0]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5, label='Died', color='red')
        axes[0, 2].set_title('Age Distribution by Survival')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].legend()
        
        # 4. Survival by Embarked
        self.train_df.groupby('Embarked')['Survived'].mean().plot(kind='bar', ax=axes[1, 0], color='purple')
        axes[1, 0].set_title('Survival Rate by Embarkation Port')
        axes[1, 0].set_ylabel('Survival Rate')
        axes[1, 0].set_xticklabels(['C', 'Q', 'S'], rotation=0)
        
        # 5. Survival by Family Size
        self.train_df.groupby('FamilySize_temp')['Survived'].mean().plot(kind='bar', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('Survival Rate by Family Size')
        axes[1, 1].set_ylabel('Survival Rate')
        axes[1, 1].set_xlabel('Family Size')
        
        # 6. Correlation heatmap
        numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        sns.heatmap(self.train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1, 2], center=0)
        axes[1, 2].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("\n[Visualization saved: eda_visualizations.png]")
        plt.close()
    
    # ============================================================================
    # STEP 3: HANDLING MISSING VALUES
    # ============================================================================
    
    def handle_missing_values(self):
        """
        Impute missing values with appropriate strategies.
        - Age: Use median by group (Pclass, Sex)
        - Embarked: Use mode
        - Cabin: Create HasCabin feature
        - Fare: Use median
        """
        print("\n" + "="*80)
        print("STEP 3: HANDLING MISSING VALUES")
        print("="*80)
        
        # Age: Use median by group (Pclass, Sex)
        print("\n--- Imputing Age ---")
        print("Strategy: Median by group (Pclass, Sex)")
        print("Justification: Age varies significantly by class and gender")
        
        # Calculate median age by Pclass and Sex from TRAINING data only
        age_median_by_group = self.train_df.groupby(['Pclass', 'Sex'])['Age'].median()
        overall_median_age = self.train_df['Age'].median()
        
        # Apply to both train and test using the same medians
        for df in [self.train_df, self.test_df]:
            # Fill missing ages based on Pclass and Sex
            for (pclass, sex), median_age in age_median_by_group.items():
                mask = (df['Pclass'] == pclass) & (df['Sex'] == sex) & (df['Age'].isnull())
                df.loc[mask, 'Age'] = median_age
            
            # If any Age still missing (edge case), fill with overall median from training
            df['Age'].fillna(overall_median_age, inplace=True)
        
        # Embarked: Use mode
        print("\n--- Imputing Embarked ---")
        print("Strategy: Mode (most common value)")
        mode_embarked = self.train_df['Embarked'].mode()[0]
        print(f"Justification: Only 2 missing values, using mode: {mode_embarked}")
        
        self.train_df['Embarked'].fillna(mode_embarked, inplace=True)
        self.test_df['Embarked'].fillna(mode_embarked, inplace=True)
        
        # Cabin: Create HasCabin feature (done in feature engineering)
        print("\n--- Handling Cabin ---")
        print("Strategy: Create binary feature 'HasCabin'")
        print("Justification: Cabin number itself not useful, but having cabin info indicates deck/status")
        
        # Fare: Use median (for test set)
        print("\n--- Imputing Fare ---")
        print("Strategy: Median (from training set)")
        fare_median = self.train_df['Fare'].median()
        # Only fill test set - train set should not have missing Fare
        self.test_df['Fare'].fillna(fare_median, inplace=True)
        print(f"Justification: Uses training set median for test imputation: {fare_median:.2f}")
        
        # Verify no missing values in key columns
        print("\n--- Verification: Missing Values After Imputation ---")
        print("Train:", self.train_df[['Age', 'Embarked', 'Fare']].isnull().sum())
        print("Test:", self.test_df[['Age', 'Embarked', 'Fare']].isnull().sum())
    
    # ============================================================================
    # STEP 4: FEATURE ENGINEERING
    # ============================================================================
    
    def feature_engineering(self):
        """
        Create new features to improve model performance.
        - FamilySize: SibSp + Parch + 1
        - IsAlone: 1 if FamilySize == 1, else 0
        - Title: Extract from Name
        - AgeGroup: Categorize Age
        - FareGroup: Categorize Fare into quartiles
        - HasCabin: 1 if Cabin not empty, else 0
        """
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)
        
        for idx, df in enumerate([self.train_df, self.test_df]):
            # FamilySize
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            if idx == 0:
                print(f"\n✓ Created FamilySize: SibSp + Parch + 1")
            
            # IsAlone
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            if idx == 0:
                print(f"✓ Created IsAlone: 1 if FamilySize == 1, else 0")
            
            # Title extraction
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            
            # Group rare titles
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
                'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
                'Capt': 'Rare', 'Ms': 'Miss'
            }
            df['Title'] = df['Title'].map(title_mapping)
            df['Title'].fillna('Rare', inplace=True)
            if idx == 0:
                print(f"✓ Created Title: Extracted from Name (Mr, Mrs, Miss, Master, Rare)")
            
            # AgeGroup - use consistent bins
            df['AgeGroup'] = pd.cut(df['Age'], 
                                    bins=self.AGE_BINS,
                                    labels=self.AGE_LABELS)
            if idx == 0:
                print(f"✓ Created AgeGroup: Categorized Age (Child, Adolescent, Adult, Middle-aged, Senior)")
            
            # FareGroup - calculate bins from training data only
            if idx == 0:
                # First iteration - training data
                df['FareGroup'], self.fare_bins = pd.qcut(df['Fare'], q=4, 
                                                          labels=['Low', 'Medium', 'High', 'Very High'], 
                                                          duplicates='drop', retbins=True)
                print(f"✓ Created FareGroup: Quartiles (Low, Medium, High, Very High)")
            else:
                # Second iteration - test data, use training bins
                df['FareGroup'] = pd.cut(df['Fare'], bins=self.fare_bins,
                                        labels=['Low', 'Medium', 'High', 'Very High'],
                                        include_lowest=True)
            
            # HasCabin
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            if idx == 0:
                print(f"✓ Created HasCabin: 1 if Cabin not empty, else 0")
        
        # Display new features
        print("\n--- Sample of Engineered Features ---")
        print(self.train_df[['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup', 'HasCabin']].head(10))
    
    # ============================================================================
    # STEP 5: ENCODING CATEGORICAL VARIABLES
    # ============================================================================
    
    def encode_features(self):
        """
        Encode categorical variables appropriately.
        - One-Hot Encoding for Sex, Embarked, Title, AgeGroup, FareGroup
        - Ensure consistent encoding for train and test
        """
        print("\n" + "="*80)
        print("STEP 5: ENCODING CATEGORICAL VARIABLES")
        print("="*80)
        
        # Combine train and test for consistent encoding
        train_len = len(self.train_df)
        combined = pd.concat([self.train_df, self.test_df], axis=0, sort=False)
        
        # One-Hot Encoding with drop_first=True to avoid multicollinearity
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        
        print(f"\n--- Applying One-Hot Encoding ---")
        print(f"Features: {categorical_features}")
        print(f"Setting drop_first=True to avoid multicollinearity")
        
        combined = pd.get_dummies(combined, columns=categorical_features, drop_first=True)
        
        # Split back to train and test
        self.train_df = combined[:train_len]
        self.test_df = combined[train_len:]
        
        print(f"\n✓ Encoding completed")
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        # Display encoded columns
        print("\n--- Encoded Columns (sample) ---")
        encoded_cols = [col for col in self.train_df.columns if any(cat in col for cat in categorical_features)]
        print(encoded_cols[:10])
    
    # ============================================================================
    # STEP 6: FEATURE SELECTION
    # ============================================================================
    
    def select_features(self):
        """
        Select relevant features for modeling.
        - Remove: PassengerId, Name, Ticket, Cabin, temporary columns
        - Keep: All engineered and encoded features
        """
        print("\n" + "="*80)
        print("STEP 6: FEATURE SELECTION")
        print("="*80)
        
        # Columns to drop
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 
                       'AgeGroup_temp', 'FamilySize_temp']
        
        # Drop from train (keep Survived)
        cols_to_drop_train = [col for col in cols_to_drop if col in self.train_df.columns]
        if 'Survived' in self.train_df.columns:
            X_train_full = self.train_df.drop(cols_to_drop_train + ['Survived'], axis=1, errors='ignore')
            y_train_full = self.train_df['Survived']
        
        # Drop from test (save PassengerId for submission)
        test_passenger_ids = self.test_df['PassengerId'].copy()
        cols_to_drop_test = [col for col in cols_to_drop if col in self.test_df.columns]
        # Also drop Survived if it exists in test (from concatenation)
        cols_to_drop_test_all = cols_to_drop_test + ['Survived']
        X_test_full = self.test_df.drop(cols_to_drop_test_all, axis=1, errors='ignore')
        
        print(f"\n--- Excluded Features ---")
        print(f"Dropped: {cols_to_drop_train}")
        
        print(f"\n--- Final Feature Set ---")
        self.feature_names = X_train_full.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Features: {self.feature_names}")
        
        # Store for later use
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_test_full = X_test_full
        self.test_passenger_ids = test_passenger_ids
        
        return X_train_full, y_train_full, X_test_full
    
    # ============================================================================
    # STEP 7: NORMALIZATION/STANDARDIZATION
    # ============================================================================
    
    def normalize_features(self):
        """
        Apply StandardScaler to numerical features.
        Rationale: Logistic Regression is sensitive to feature scales.
        Ensures all features contribute equally to the distance calculations.
        """
        print("\n" + "="*80)
        print("STEP 7: NORMALIZATION/STANDARDIZATION")
        print("="*80)
        
        print("\n--- Rationale ---")
        print("• Logistic Regression uses gradient descent, which converges faster with normalized features")
        print("• Features with larger scales can dominate the optimization process")
        print("• StandardScaler ensures all features have mean=0 and std=1")
        print("• Regularization (L1/L2) works better when features are on same scale")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Fit on training data and transform both train and test
        self.X_train_full_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train_full),
            columns=self.X_train_full.columns,
            index=self.X_train_full.index
        )
        
        self.X_test_full_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test_full),
            columns=self.X_test_full.columns,
            index=self.X_test_full.index
        )
        
        print(f"\n✓ StandardScaler fitted on training data")
        print(f"✓ Applied to both training and test sets")
        print(f"\n--- Sample Statistics After Scaling ---")
        print(f"Mean: {self.X_train_full_scaled.mean().mean():.6f} (should be ≈ 0)")
        print(f"Std: {self.X_train_full_scaled.std().mean():.6f} (should be ≈ 1)")
    
    # ============================================================================
    # STEP 8: DATA SPLITTING
    # ============================================================================
    
    def split_data(self):
        """
        Split training data into train/validation sets (80/20).
        Use random_state=42 for reproducibility.
        Validate target distribution consistency.
        """
        print("\n" + "="*80)
        print("STEP 8: DATA SPLITTING")
        print("="*80)
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_full_scaled,
            self.y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train_full  # Ensures balanced distribution
        )
        
        print(f"\n--- Split Configuration ---")
        print(f"Train/Validation split: 80/20")
        print(f"Random state: 42 (for reproducibility)")
        print(f"Stratified: Yes (maintains target distribution)")
        
        print(f"\n--- Dataset Sizes ---")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Validation set: {len(self.X_val)} samples")
        
        print(f"\n--- Target Distribution Validation ---")
        print(f"Full dataset - Survived: {self.y_train_full.mean():.4f}")
        print(f"Training set - Survived: {self.y_train.mean():.4f}")
        print(f"Validation set - Survived: {self.y_val.mean():.4f}")
        print(f"✓ Distribution is consistent across splits")
    
    # ============================================================================
    # STEP 9: BASE MODEL TRAINING
    # ============================================================================
    
    def train_baseline_model(self):
        """
        Train a baseline Logistic Regression model.
        - Fit on training split
        - Validate on validation split
        - Evaluate predictive performance
        """
        print("\n" + "="*80)
        print("STEP 9: BASE MODEL TRAINING")
        print("="*80)
        
        print("\n--- Training Baseline Logistic Regression ---")
        print("Configuration: Default hyperparameters, max_iter=1000")
        
        # Initialize and train baseline model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        
        print(f"\n--- Baseline Model Performance ---")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Difference: {abs(train_accuracy - val_accuracy):.4f}")
        
        if abs(train_accuracy - val_accuracy) < 0.05:
            print("✓ Model shows good generalization (low overfitting)")
        elif train_accuracy > val_accuracy:
            print("⚠ Possible overfitting detected")
        else:
            print("⚠ Unusual pattern: validation > training")
        
        return self.model
    
    # ============================================================================
    # STEP 10: MODEL EVALUATION
    # ============================================================================
    
    def evaluate_model(self, model=None, model_name="Baseline Model"):
        """
        Comprehensive model evaluation.
        - Train/Validation accuracy
        - Confusion Matrix
        - Classification Report
        - ROC curve and AUC
        """
        print("\n" + "="*80)
        print(f"STEP 10: MODEL EVALUATION - {model_name}")
        print("="*80)
        
        if model is None:
            model = self.model
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_val_pred = model.predict(self.X_val)
        y_val_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Accuracy
        train_acc = accuracy_score(self.y_train, y_train_pred)
        val_acc = accuracy_score(self.y_val, y_val_pred)
        
        print(f"\n--- Accuracy Scores ---")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Confusion Matrix
        print(f"\n--- Confusion Matrix (Validation Set) ---")
        cm = confusion_matrix(self.y_val, y_val_pred)
        print(cm)
        print(f"\nInterpretation:")
        print(f"  True Negatives (TN): {cm[0,0]} | False Positives (FP): {cm[0,1]}")
        print(f"  False Negatives (FN): {cm[1,0]} | True Positives (TP): {cm[1,1]}")
        
        # Classification Report
        print(f"\n--- Classification Report (Validation Set) ---")
        print(classification_report(self.y_val, y_val_pred, 
                                   target_names=['Died', 'Survived']))
        
        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(self.y_val, y_val_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n--- ROC AUC Score ---")
        print(f"AUC: {roc_auc:.4f}")
        
        # Plot ROC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        print(f"\n[Visualization saved: roc_curve_{model_name.replace(' ', '_').lower()}.png]")
        plt.close()
        
        # Overfitting/Underfitting Analysis
        print(f"\n--- Overfitting/Underfitting Analysis ---")
        diff = train_acc - val_acc
        if diff < 0.02:
            print("✓ Well-fitted model (minimal overfitting)")
        elif diff < 0.05:
            print("⚠ Slight overfitting detected")
        else:
            print("⚠⚠ Significant overfitting - consider regularization")
        
        if val_acc < 0.70:
            print("⚠ Low validation accuracy - possible underfitting")
        
        return val_acc, roc_auc
    
    # ============================================================================
    # STEP 11: HYPERPARAMETER OPTIMIZATION
    # ============================================================================
    
    def optimize_hyperparameters(self):
        """
        Optimize hyperparameters using GridSearchCV.
        - Parameters: C, penalty, solver
        - 5-fold cross-validation
        - Refit with best parameters
        """
        print("\n" + "="*80)
        print("STEP 11: HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        print("\n--- Grid Search Configuration ---")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        print(f"Parameters to optimize:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nCross-validation folds: 5")
        print(f"Scoring metric: accuracy")
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("\n--- Running Grid Search (this may take a moment) ---")
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\n--- Optimization Results ---")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Refit model with best parameters
        self.best_model = grid_search.best_estimator_
        
        print(f"\n✓ Model refitted with optimal hyperparameters")
        
        return self.best_model
    
    # ============================================================================
    # STEP 12: CROSS-VALIDATION
    # ============================================================================
    
    def cross_validate_model(self):
        """
        Validate optimized model using StratifiedKFold.
        - 5-fold cross-validation
        - Report mean accuracy and standard deviation
        - Compare with baseline
        """
        print("\n" + "="*80)
        print("STEP 12: CROSS-VALIDATION")
        print("="*80)
        
        print("\n--- Stratified K-Fold Cross-Validation ---")
        print("Configuration: 5 folds, stratified sampling")
        
        # Cross-validation with optimized model
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores_optimized = cross_val_score(
            self.best_model, 
            self.X_train_full_scaled, 
            self.y_train_full,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Cross-validation with baseline model
        cv_scores_baseline = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            self.X_train_full_scaled,
            self.y_train_full,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"\n--- Cross-Validation Results ---")
        print(f"\nOptimized Model:")
        print(f"  Mean Accuracy: {cv_scores_optimized.mean():.4f}")
        print(f"  Standard Deviation: {cv_scores_optimized.std():.4f}")
        print(f"  Scores by fold: {[f'{score:.4f}' for score in cv_scores_optimized]}")
        
        print(f"\nBaseline Model:")
        print(f"  Mean Accuracy: {cv_scores_baseline.mean():.4f}")
        print(f"  Standard Deviation: {cv_scores_baseline.std():.4f}")
        
        print(f"\n--- Comparison ---")
        improvement = cv_scores_optimized.mean() - cv_scores_baseline.mean()
        print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        if improvement > 0.01:
            print("✓ Hyperparameter optimization improved model performance")
        elif improvement > 0:
            print("✓ Slight improvement from optimization")
        else:
            print("⚠ No significant improvement (baseline already optimal)")
        
        return cv_scores_optimized
    
    # ============================================================================
    # STEP 13: COEFFICIENT ANALYSIS
    # ============================================================================
    
    def analyze_coefficients(self):
        """
        Interpret regression coefficients.
        - Identify most impactful features
        - Analyze feature influence on predictions
        """
        print("\n" + "="*80)
        print("STEP 13: COEFFICIENT ANALYSIS")
        print("="*80)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.best_model.coef_[0]
        })
        
        # Sort by absolute value
        coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        print("\n--- Top 10 Most Impactful Features ---")
        print(coefficients[['Feature', 'Coefficient']].head(10).to_string(index=False))
        
        print("\n--- Interpretation ---")
        print("\nPositive coefficients increase survival probability:")
        positive_coef = coefficients[coefficients['Coefficient'] > 0].head(5)
        for idx, row in positive_coef.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        print("\nNegative coefficients decrease survival probability:")
        negative_coef = coefficients[coefficients['Coefficient'] < 0].head(5)
        for idx, row in negative_coef.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        top_features = coefficients.head(15)
        colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top 15 Feature Coefficients (Logistic Regression)')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
        print("\n[Visualization saved: feature_coefficients.png]")
        plt.close()
        
        return coefficients
    
    # ============================================================================
    # STEP 14: TEST SET PREDICTIONS
    # ============================================================================
    
    def generate_predictions(self):
        """
        Apply preprocessing and model to test set.
        - Ensure no missing values
        - Generate predictions
        - Save to submission.csv
        """
        print("\n" + "="*80)
        print("STEP 14: TEST SET PREDICTIONS")
        print("="*80)
        
        # Verify no missing values
        print("\n--- Verification: Missing Values in Test Set ---")
        missing_test = self.X_test_full_scaled.isnull().sum().sum()
        print(f"Total missing values: {missing_test}")
        
        if missing_test > 0:
            print("⚠ Missing values found - this should not happen after proper preprocessing")
            print("Missing values by column:")
            print(self.X_test_full_scaled.isnull().sum()[self.X_test_full_scaled.isnull().sum() > 0])
            raise ValueError("Missing values detected in test set after preprocessing")
        
        # Generate predictions using optimized model
        print("\n--- Generating Predictions ---")
        predictions = self.best_model.predict(self.X_test_full_scaled)
        
        # Create submission file
        submission = pd.DataFrame({
            'PassengerId': self.test_passenger_ids,
            'Survived': predictions.astype(int)
        })
        
        submission.to_csv('submission.csv', index=False)
        
        print(f"✓ Predictions generated for {len(predictions)} passengers")
        print(f"✓ Submission file saved: submission.csv")
        
        print(f"\n--- Prediction Statistics ---")
        print(f"Predicted Survived: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
        print(f"Predicted Died: {len(predictions) - predictions.sum()} ({(len(predictions) - predictions.sum())/len(predictions)*100:.2f}%)")
        
        print("\n--- Sample Predictions ---")
        print(submission.head(10))
        
        return submission
    
    # ============================================================================
    # STEP 15: RECOMMENDATIONS FOR IMPROVEMENT
    # ============================================================================
    
    def provide_recommendations(self):
        """
        Provide suggestions for improving the model.
        """
        print("\n" + "="*80)
        print("STEP 15: RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*80)
        
        recommendations = """
        
1. ADVANCED FEATURE ENGINEERING
   • Interaction features: Pclass × Sex, Age × Pclass
   • Polynomial features for numerical variables
   • Extract more granular information from Cabin (deck letter)
   • Create fare-per-person feature (Fare / FamilySize)
   • Binning: Experiment with different age/fare grouping strategies
   
2. OUTLIER DETECTION AND TREATMENT
   • Identify outliers in Fare and Age using IQR or z-score
   • Analyze impact of extreme values on model performance
   • Consider robust scaling methods (RobustScaler)
   • Investigate passengers with very high fares or unusual ages
   
3. ALTERNATIVE ALGORITHMS
   • Random Forest: Handles non-linear relationships, feature interactions
   • Gradient Boosting (XGBoost, LightGBM, CatBoost): Often best performance
   • Support Vector Machines: Effective for binary classification
   • Neural Networks: Can capture complex patterns
   • Naive Bayes: Fast baseline, works well with categorical features
   
4. ENSEMBLE TECHNIQUES
   • Voting Classifier: Combine multiple models (Logistic, RF, XGBoost)
   • Stacking: Use meta-learner on predictions from base models
   • Bagging: Bootstrap aggregating for variance reduction
   • Boosting: Sequential learning to improve weak learners
   
5. FEATURE SELECTION TECHNIQUES
   • Recursive Feature Elimination (RFE)
   • SelectKBest with chi-squared or f_classif
   • L1 regularization for automatic feature selection
   • Principal Component Analysis (PCA) for dimensionality reduction
   
6. HYPERPARAMETER TUNING
   • Use RandomizedSearchCV for larger parameter spaces
   • Bayesian optimization (e.g., Optuna, Hyperopt)
   • Try different regularization strengths more granularly
   • Experiment with class_weight for imbalanced data handling
   
7. CROSS-VALIDATION STRATEGIES
   • Nested cross-validation for unbiased evaluation
   • Leave-one-out cross-validation for small datasets
   • Time-based splits if temporal patterns exist
   
8. DATA AUGMENTATION
   • SMOTE for balancing classes (if needed)
   • Generate synthetic samples using mixup
   
9. MODEL INTERPRETABILITY
   • SHAP values for feature importance
   • LIME for local interpretability
   • Partial dependence plots
   
10. ADDITIONAL DATA SOURCES
    • External datasets: Historical context, ship layout
    • Domain knowledge: Titanic disaster research
    • Feature engineering based on survival stories
        """
        
        print(recommendations)
    
    # ============================================================================
    # MAIN EXECUTION METHOD
    # ============================================================================
    
    def run_complete_analysis(self):
        """
        Execute all 15 steps in sequence.
        Complete end-to-end solution for Titanic classification.
        """
        print("\n" + "="*80)
        print("TITANIC SURVIVAL PREDICTION - COMPLETE SOLUTION")
        print("Systematic Approach using Logistic Regression")
        print("="*80)
        
        # Execute all steps
        self.load_and_explore()
        self.exploratory_analysis()
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_features()
        self.select_features()
        self.normalize_features()
        self.split_data()
        self.train_baseline_model()
        self.evaluate_model(self.model, "Baseline Model")
        self.optimize_hyperparameters()
        self.evaluate_model(self.best_model, "Optimized Model")
        self.cross_validate_model()
        self.analyze_coefficients()
        self.generate_predictions()
        self.provide_recommendations()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  • submission.csv - Kaggle submission file")
        print("  • eda_visualizations.png - EDA charts")
        print("  • roc_curve_baseline_model.png - ROC curve for baseline")
        print("  • roc_curve_optimized_model.png - ROC curve for optimized model")
        print("  • feature_coefficients.png - Feature importance visualization")
        print("\n" + "="*80)


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize and run complete analysis
    solution = TitanicSolution(
        train_path='data/train.csv',
        test_path='data/test.csv'
    )
    
    solution.run_complete_analysis()
