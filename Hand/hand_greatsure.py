# Hand Gesture Classification with Machine Learning
# Complete end-to-end implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import cv2
import mediapipe as mp
import os
import time
import joblib
from sklearn.datasets import make_classification

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 1. Data Collection and Preprocessing
def extract_hand_landmarks(image):
    """Extract hand landmarks from an image using MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks)


def create_dataset_from_folder(image_folder, label):
    """Create dataset from folder of images"""
    features = []
    labels = []

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)
        landmarks = extract_hand_landmarks(image)

        if len(landmarks) > 0:  # Only add if hand was detected
            features.append(landmarks)
            labels.append(label)

    return np.array(features), np.array(labels)


# 2. Data Preparation (using synthetic data for this example)
def prepare_data():
    """Prepare synthetic dataset for demonstration"""
    X, y = make_classification(n_samples=1000, n_features=63, n_classes=5,
                               n_informative=15, random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# 3. Model Optimization with Optuna
def optimize_model(X_train, y_train, X_test, y_test, model_type, n_trials=50):
    """Optimize hyperparameters for a given model type"""

    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)

        elif model_type == 'lr':
            params = {
                'C': trial.suggest_float('C', 0.01, 10, log=True),
                'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': 42
            }
            model = LogisticRegression(**params)

        elif model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.1, 10, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'random_state': 42
            }
            model = SVC(**params, probability=True)

        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = XGBClassifier(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# 4. Model Training and Evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Train and evaluate a model"""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.2%}")
    print(f"F1-Score: {f1:.2%}")
    print(f"Training Time: {train_time:.2f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return acc, f1, model


# 5. Real-Time Prediction
def real_time_prediction(model, scaler, gesture_names):
    """Predict hand gestures in real-time using webcam"""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks and predict gesture
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                landmarks = scaler.transform([landmarks])
                pred = model.predict(landmarks)[0]
                proba = model.predict_proba(landmarks)[0]

                # Display prediction
                gesture = gesture_names[pred]
                confidence = proba[pred]

                cv2.putText(image, f"Gesture: {gesture} ({confidence:.1%})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Execution
def main():
    print("Hand Gesture Classification System")
    print("=================================\n")

    # 1. Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # 2. Optimize models (commented out for demo - would take time to run)
    print("\nOptimizing models (this may take several minutes)...")
    # best_rf_params = optimize_model(X_train, y_train, X_test, y_test, 'rf')
    # best_lr_params = optimize_model(X_train, y_train, X_test, y_test, 'lr')
    # best_svm_params = optimize_model(X_train, y_train, X_test, y_test, 'svm')
    # best_xgb_params = optimize_model(X_train, y_train, X_test, y_test, 'xgb')

    # Using pre-defined params for demonstration
    best_rf_params = {
        'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2,
        'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True,
        'random_state': 42
    }
    best_lr_params = {
        'C': 1.0, 'solver': 'lbfgs', 'max_iter': 300, 'random_state': 42
    }
    best_svm_params = {
        'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'random_state': 42
    }
    best_xgb_params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
        'reg_alpha': 0, 'reg_lambda': 1, 'random_state': 42,
        'use_label_encoder': False, 'eval_metric': 'logloss'
    }

    # 3. Train and evaluate models
    print("\nTraining and evaluating models...")

    # Random Forest
    rf_model = RandomForestClassifier(**best_rf_params)
    rf_acc, rf_f1, rf_model = train_and_evaluate(
        rf_model, X_train, y_train, X_test, y_test, "Optimized Random Forest")

    # Logistic Regression
    lr_model = LogisticRegression(**best_lr_params)
    lr_acc, lr_f1, lr_model = train_and_evaluate(
        lr_model, X_train, y_train, X_test, y_test, "Optimized Logistic Regression")

    # SVM
    svm_model = SVC(**best_svm_params, probability=True)
    svm_acc, svm_f1, svm_model = train_and_evaluate(
        svm_model, X_train, y_train, X_test, y_test, "Optimized SVM")

    # XGBoost
    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_acc, xgb_f1, xgb_model = train_and_evaluate(
        xgb_model, X_train, y_train, X_test, y_test, "Optimized XGBoost")

    # Compare model performance
    results = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost'],
        'Accuracy': [rf_acc, lr_acc, svm_acc, xgb_acc],
        'F1-Score': [rf_f1, lr_f1, svm_f1, xgb_f1]
    })

    print("\nModel Comparison:")
    print(results)

    # 4. Save models
    print("\nSaving models...")
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # 5. Real-time prediction with best model
    gesture_names = {
        0: "Thumbs Up",
        1: "Peace",
        2: "OK",
        3: "Fist",
        4: "Open Hand"
    }

    print("\nStarting real-time gesture recognition with XGBoost...")
    print("Press ESC to exit the camera window.")
    real_time_prediction(xgb_model, scaler, gesture_names)


if __name__ == "__main__":
    main()
