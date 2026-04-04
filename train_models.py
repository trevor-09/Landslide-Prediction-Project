import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Setup paths
DATA_DIR = r"c:\Users\annaa\Downloads\Landslide"
STATIC_DIR = os.path.join(DATA_DIR, "static")
MODEL_DIR = os.path.join(DATA_DIR, "models")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_combine(place_name, file_configs):
    combined_data = []
    for file_name, angle in file_configs:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            # Add Angle column if not already or if needs override
            df['Angle'] = angle
            combined_data.append(df)
    
    if not combined_data:
        return None
    
    final_df = pd.concat(combined_data, ignore_index=True)
    return final_df

def preprocess_data(df):
    # Ignore DateTime columns
    cols_to_drop = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    df = df.drop(columns=cols_to_drop)
    
    # Handle missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

from sklearn.tree import plot_tree

def train_ann(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return model, y_pred, history

def evaluate_models(place_name, X_train, X_test, y_train, y_test, le):
    results = []
    num_classes = len(np.unique(y_test))
    
    # 1. ANN
    ann_model, ann_y_pred, ann_history = train_ann(X_train, y_train, X_test, y_test, num_classes)
    
    # Save ANN History Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(ann_history.history['accuracy'], label='Train Accuracy')
    plt.plot(ann_history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'ANN Accuracy ({place_name})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(ann_history.history['loss'], label='Train Loss')
    plt.plot(ann_history.history['val_loss'], label='Val Loss')
    plt.title(f'ANN Loss ({place_name})')
    plt.legend()
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_ann_history.png'))
    plt.close()
    
    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_y_pred = dt_model.predict(X_test)
    
    # Save Decision Tree Plot
    plt.figure(figsize=(20,10))
    plot_tree(dt_model, feature_names=X_train.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree Logic ({place_name})')
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_decision_tree_logic.png'))
    plt.close()
    
    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    
    # Save Random Forest Feature Importance
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=X_train.columns)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title(f'Feature Importance: Random Forest ({place_name})')
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_random_forest_importance.png'))
    plt.close()
    
    # 4. KNN
    knn_scores = []
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        knn_scores.append(score)
    
    plt.figure(figsize=(10,6))
    plt.plot(range(1, 21), knn_scores, marker='o')
    plt.title(f'KNN Accuracy vs. Neighbors ({place_name})')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_knn_accuracy_k.png'))
    plt.close()
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_y_pred = knn_model.predict(X_test)
    
    models = {
        "ANN": (ann_model, ann_y_pred),
        "Decision Tree": (dt_model, dt_y_pred),
        "Random Forest": (rf_model, rf_y_pred),
        "KNN": (knn_model, knn_y_pred)
    }
    
    for name, (model, y_pred) in models.items():
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cv_score = 0
        if name != "ANN":
              cv_score = cross_val_score(model, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), cv=5).mean()
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "CV Score": cv_score
        })
        
        # Save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix: {name} ({place_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_{name.lower().replace(" ", "_")}_cm.png'))
        plt.close()
        
        # Save Classification Report
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        with open(os.path.join(DATA_DIR, f'{place_name}_{name.lower().replace(" ", "_")}_report.txt'), 'w') as f:
            f.write(report)

    # Best model selection
    best_model_info = max(results, key=lambda x: x['Accuracy'])
    best_model_name = best_model_info['Model']
    best_model = models[best_model_name][0]
    
    # Save best model
    if best_model_name == "ANN":
        best_model.save(os.path.join(MODEL_DIR, f'{place_name}_best_model.h5'))
    else:
        joblib.dump(best_model, os.path.join(MODEL_DIR, f'{place_name}_best_model.pkl'))
    
    return results, best_model_name, le

def run_pipeline(place_name, file_configs):
    print(f"\nProcessing {place_name}...")
    df = load_and_combine(place_name, file_configs)
    if df is None:
        print(f"No data for {place_name}")
        return
    
    df = preprocess_data(df)
    
    # EDA Plots
    plt.figure(figsize=(8,6))
    sns.countplot(x='Category', data=df)
    plt.title(f'Category Distribution ({place_name})')
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_category_dist.png'))
    plt.close()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12,10))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap ({place_name})')
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_heatmap.png'))
    plt.close()
    
    # t-SNE Projection
    from sklearn.manifold import TSNE
    X_for_tsne = df.drop(columns=['Category']).select_dtypes(include=[np.number])
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_for_tsne)
    
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df['Category'], palette='bright', s=60, alpha=0.7)
    plt.legend(title='Ground Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE Data Projection ({place_name})')
    plt.xlabel('t-SNE Axis 1')
    plt.ylabel('t-SNE Axis 2')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f'{place_name}_tsne.png'))
    plt.close()
    
    # Encoding and Splitting
    le = LabelEncoder()
    df['Category'] = le.fit_transform(df['Category'])
    joblib.dump(le, os.path.join(MODEL_DIR, f'{place_name}_label_encoder.pkl'))
    
    X = df.drop(columns=['Category'])
    y = df['Category']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{place_name}_scaler.pkl'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results, best_model_name, le = evaluate_models(place_name, X_train, X_test, y_train, y_test, le)
    
    # Save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(DATA_DIR, f'{place_name}_comparison.csv'), index=False)
    
    print(f"Finished {place_name}. Best Model: {best_model_name}")

if __name__ == "__main__":
    # Veeramalakunnu (Place 1)
    run_pipeline("Veeramalakunnu", [("place1_30.xlsx", 30), ("place1_60.xlsx", 60)])
    
    # Thekkil (Place 2)
    run_pipeline("Thekkil", [("Place2_30.xlsx", 30), ("Place2_60.xlsx", 60)])
