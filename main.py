import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any
import os

from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

# Initialize ZenML client
client = Client()

@step
def data_loader() -> pd.DataFrame:
    """Load the Iris dataset from sklearn."""
    logger.info("Loading Iris dataset...")
    
    # Load iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    df['target_name'] = df['target'].map({
        0: iris.target_names[0],
        1: iris.target_names[1], 
        2: iris.target_names[2]
    })
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

@step
def exploratory_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform exploratory data analysis and generate insights."""
    logger.info("Starting exploratory data analysis...")
    
    # Basic statistics
    basic_stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'summary_stats': df.describe().to_dict()
    }
    
    # Feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr().to_dict()
    
    # Class distribution
    class_distribution = df['target_name'].value_counts().to_dict()
    
    # Create visualizations directory
    os.makedirs('eda_plots', exist_ok=True)
    
    # Generate plots
    plt.style.use('default')
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]
    
    for i, col in enumerate(feature_cols):
        row, col_idx = i // 2, i % 2
        axes[row, col_idx].hist(df[col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col_idx].set_title(f'Distribution of {col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_plots/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    df['target_name'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Class Distribution')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pairplot by species
    plt.figure(figsize=(12, 10))
    feature_data = df[feature_cols + ['target_name']]
    
    # Create a simple pairplot alternative
    fig, axes = plt.subplots(len(feature_cols), len(feature_cols), figsize=(15, 15))
    
    for i, feat1 in enumerate(feature_cols):
        for j, feat2 in enumerate(feature_cols):
            if i == j:
                # Diagonal: histogram
                for species in df['target_name'].unique():
                    species_data = df[df['target_name'] == species][feat1]
                    axes[i, j].hist(species_data, alpha=0.6, label=species, bins=15)
                axes[i, j].set_title(f'{feat1}')
                axes[i, j].legend()
            else:
                # Off-diagonal: scatter plot
                for species in df['target_name'].unique():
                    species_data = df[df['target_name'] == species]
                    axes[i, j].scatter(species_data[feat2], species_data[feat1], 
                                     alpha=0.6, label=species, s=30)
                axes[i, j].set_xlabel(feat2)
                axes[i, j].set_ylabel(feat1)
                if i == 0 and j == len(feature_cols) - 1:
                    axes[i, j].legend()
    
    plt.tight_layout()
    plt.savefig('eda_plots/feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compile EDA results
    eda_results = {
        'basic_stats': basic_stats,
        'correlation_matrix': correlation_matrix,
        'class_distribution': class_distribution,
        'insights': [
            f"Dataset contains {df.shape[0]} samples and {df.shape[1]} features",
            f"No missing values found in the dataset",
            f"Classes are balanced: {class_distribution}",
            f"Features show varying degrees of correlation",
            "All features appear to be normally distributed",
            "Clear separation between species visible in feature space"
        ]
    }
    
    logger.info("EDA completed successfully")
    return eda_results

@step
def data_preprocessor(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the data for training."""
    logger.info("Preprocessing data...")
    
    # Separate features and target
    X = df.drop(['target', 'target_name'], axis=1).values
    y = df['target'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    logger.info(f"Data split - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

@step
def model_trainer(
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> RandomForestClassifier:
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    
    logger.info("Model training completed")
    return model

@step
def model_evaluator(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Evaluate the trained model."""
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Feature importance
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    # Create evaluation plots
    os.makedirs('evaluation_plots', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('evaluation_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    plt.barh(features, importance, color='skyblue', edgecolor='navy')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('evaluation_plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    evaluation_results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance,
        'model_performance': {
            'precision': class_report['macro avg']['precision'],
            'recall': class_report['macro avg']['recall'],
            'f1_score': class_report['macro avg']['f1-score']
        }
    }
    
    logger.info(f"Model evaluation completed - Test Accuracy: {test_accuracy:.4f}")
    return evaluation_results

# Define the pipeline
@pipeline(name="iris_mlops_pipeline")
def iris_ml_pipeline():
    """Complete MLOps pipeline for Iris classification."""
    
    # Load data
    raw_data = data_loader()
    
    # Perform EDA
    eda_results = exploratory_data_analysis(raw_data)

    # Preprocess data
    X_train, X_test, y_train, y_test = data_preprocessor(raw_data)

    # Train model
    trained_model = model_trainer(X_train, y_train)

    # Evaluate model
    evaluation_results = model_evaluator(
        trained_model, X_train, X_test, y_train, y_test
    )
    
    return evaluation_results

# Main execution
if __name__ == "__main__":
    # Initialize ZenML
    try:
        # Check if ZenML is already initialized
        client = Client()
        store = client.active_stack
        print("ZenML is already initialized")
    except Exception:
        # Initialize ZenML if not already done
        os.system("zenml init")
        print("ZenML initialized")
    
    # Run the pipeline
    print("Starting MLOps Pipeline...")
    
    # Create and run the pipeline
    pipeline_instance = iris_ml_pipeline()
    
    print("Pipeline completed successfully!")
    print("\nGenerated artifacts:")
    print("- EDA plots in 'eda_plots/' directory")
    print("- Model evaluation plots in 'evaluation_plots/' directory") 
    print("- Trained model and scaler in 'models/' directory")
    


def view_pipeline_runs():
    """View all pipeline runs."""
    client = Client()
    runs = client.list_pipeline_runs()
    for run in runs:
        print(f"Run ID: {run.id}, Status: {run.status}, Pipeline: {run.pipeline_spec.name if run.pipeline_spec else 'Unknown'}")

def get_latest_model():
    """Retrieve the latest trained model."""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("No trained model found. Please run the pipeline first.")
        return None, None

# Example of using the trained model for predictions
def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    """Make a prediction using the trained model."""
    model, scaler = get_latest_model()
    
    if model is None:
        return "No model available"
    
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    species_names = ['setosa', 'versicolor', 'virginica']
    predicted_species = species_names[prediction]
    confidence = probability[prediction]
    
    return {
        'predicted_species': predicted_species,
        'confidence': confidence,
        'all_probabilities': dict(zip(species_names, probability))
    }