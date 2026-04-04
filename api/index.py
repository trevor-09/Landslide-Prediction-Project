from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

# In Vercel, paths are relative to the root project directory
# Since this file is in /api/index.py, we need to point to the folders in the root.
app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')

# Relative paths for deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Simplified English explanations for Project Report (professional and easy)
EXPLANATIONS = {
    'general': {
        'distribution': {
            'title': '1. Data Quantity Analysis (Category Distribution)',
            'purpose': 'To check if we have enough examples of each ground situation.',
            'text': 'This graph counts how many times we recorded "Safe" ground versus "Failure" ground. It ensures our AI has enough examples to learn from. If one category is too small, the AI might not recognize it in real life.',
            'report_summary': 'This chart shows the distribution of our training data. A balanced chart means the AI is trained equally on all possible landslide situations.'
        },
        'correlation': {
            'title': '2. Sensor Relationship Map (Correlation Heatmap)',
            'purpose': 'To see how different sensors like Temperature and Moisture affect each other.',
            'text': 'This map uses numbers to show connections. A high number means two sensors usually go up at the same time. This helps us understand which sensors are "Partners" in detecting a landslide.',
            'report_summary': 'The heatmap identifies strong relationships between sensor features, allowing us to see which triggers are naturally linked.'
        },
        'tsne': {
            'title': '3. Data Cluster Map (t-SNE Projection)',
            'purpose': 'To visually prove that "Safe" data looks different from "Failure" data.',
            'text': 'This map squashes all sensor data into a 2D view. Each dot is one record. Look for groups of the same color! If dots of the same color stay together in their own "Island", it means the AI can easily tell them apart.',
            'report_summary': 'The t-SNE projection shows how the data naturally clusters. Distinct groups indicate that the sensor patterns are unique and predictable.'
        }
    },
    'metrics': {
        'accuracy': {
            'name': 'Accuracy',
            'desc': 'The overall percentage of correct guesses. 90% Accuracy means the AI was right 9 out of 10 times.'
        },
        'precision': {
            'name': 'Precision (Alarm Reliability)',
            'desc': 'This tells us if the "Danger" alarm is trustable. High precision means when the alarm rings, there is almost always a real risk.'
        },
        'recall': {
            'name': 'Recall (Detection Power)',
            'desc': 'This tells us if the AI missed any real landslides. High recall means the AI is very good at catching every single danger.'
        },
        'f1': {
            'name': 'F1-Score (Total Quality)',
            'desc': 'This is a single combined score that balances both reliability and detection power.'
        }
    },
    'models': [
        {
            'id': 'ann',
            'name': 'Artificial Neural Network',
            'working': 'This model works like a human brain. It uses many layers of artificial "neurons" to find hidden patterns that other models might miss.',
            'plots': [
                {
                    'title': 'Training Success History',
                    'purpose': 'To show how the AI improved its skills during the training process.',
                    'desc': 'As the lines go up, it means the brain is getting smarter. As the error (loss) goes down, it means the brain is making fewer mistakes.',
                    'filename_suffix': 'ann_history.png'
                },
                {
                    'title': 'Success Grid (Confusion Matrix)',
                    'purpose': 'To see exactly where the brain was correct or confused.',
                    'desc': 'The diagonal boxes show correct predictions. Other boxes show where the AI mixed up one ground type with another.',
                    'filename_suffix': 'ann_cm.png'
                }
            ]
        },
        {
            'id': 'decision_tree',
            'name': 'Decision Tree',
            'working': 'This model works by asking a list of "Yes or No" questions about the sensors. It follows these questions until it reaches a final answer.',
            'plots': [
                {
                    'title': 'Step-by-Step Logic Diagram',
                    'purpose': 'To show the actual questions the AI asks to reach a conclusion.',
                    'desc': 'Every box is a question (for example: "Is Moisture more than 40%?"). Following the arrows shows the path the AI takes to make a prediction.',
                    'filename_suffix': 'decision_tree_logic.png'
                },
                {
                    'title': 'Success Grid (Confusion Matrix)',
                    'purpose': 'To see the final success rate for every category.',
                    'desc': 'This grid shows how accurately the Decision Tree classified the ground conditions.',
                    'filename_suffix': 'decision_tree_cm.png'
                }
            ]
        },
        {
            'id': 'random_forest',
            'name': 'Random Forest',
            'working': 'This model uses hundreds of different Decision Trees and combines their answers. This makes it more stable and less likely to make a mistake.',
            'plots': [
                {
                    'title': 'Critical Sensor Ranking',
                    'purpose': 'To identify which sensors are the most important for predicting a landslide.',
                    'desc': 'The longest bars represent the sensors that are the "main triggers". These are the most important sensors for our safety system.',
                    'filename_suffix': 'random_forest_importance.png'
                },
                {
                    'title': 'Success Grid (Confusion Matrix)',
                    'purpose': 'To show the final reliability of the combined forest.',
                    'desc': 'This shows the very high success rate achieved by combining many individual trees.',
                    'filename_suffix': 'random_forest_cm.png'
                }
            ]
        },
        {
            'id': 'knn',
            'name': 'K-Nearest Neighbors',
            'working': 'This model looks at the most similar historical data points from the past. It assumes that if the sensor values match a past event, the result will be the same.',
            'plots': [
                {
                    'title': 'Best Match Comparison Graph',
                    'purpose': 'To show how the model selects the best number of neighbors to compare.',
                    'desc': 'The peak of this graph shows the ideal number of past "Neighbors" the AI should look at to be most accurate.',
                    'filename_suffix': 'knn_accuracy_k.png'
                },
                {
                    'title': 'Success Grid (Confusion Matrix)',
                    'purpose': 'To show how well the AI matched new data to past events.',
                    'desc': 'This grid shows how often the "Nearest Neighbor" logic was correct.',
                    'filename_suffix': 'knn_cm.png'
                }
            ]
        }
    ]
}

# Strengths of each model for the Final Project Report
MODEL_STRENGTHS = {
    'ANN': {
        'title': 'Superior Pattern Recognition',
        'reason': 'Artificial Neural Networks are modeled after the human brain. They excel at finding complex, non-linear relationships between sensors like soil moisture and humidity, making them highly reliable for deep data patterns.'
    },
    'Random Forest': {
        'title': 'Balanced Ensemble Reliability',
        'reason': 'Random Forest combines the votes of hundreds of different decision trees. This vote-averaging makes it very stable and excellent at avoiding false alarms while maintaining high detection accuracy.'
    },
    'Decision Tree': {
        'title': 'Direct Logic and Transparency',
        'reason': 'Decision Trees are best for scenarios where we need to know exactly why a decision was made. Their clear "rule-based" thinking makes them easy to audit for safety protocols.'
    },
    'KNN': {
        'title': 'Historical Comparison Accuracy',
        'reason': 'K-Nearest Neighbors is perfect for matching current sensor data to similar historical events. It is highly intuitive as it relies on past landslide patterns to predict the future.'
    }
}

def get_model_info(place_name):
    comparison_path = os.path.join(BASE_DIR, f'{place_name}_comparison.csv')
    df = pd.read_csv(comparison_path)
    
    # Selection Logic: Sort by Accuracy, then F1-Score to break ties
    df_sorted = df.sort_values(by=['Accuracy', 'F1-Score'], ascending=False)
    best_row = df_sorted.iloc[0]
    
    best_model_name = best_row['Model']
    model_info = MODEL_STRENGTHS.get(best_model_name, {'title': 'Highest Accuracy', 'reason': 'This model showed the best overall performance.'})
    best_model_reason = f"**{model_info['title']}**: {model_info['reason']}"
    
    return df.to_dict('records'), best_model_name, best_model_reason

def load_predictor(place_name, model_name):
    scaler = joblib.load(os.path.join(MODEL_DIR, f'{place_name}_scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, f'{place_name}_label_encoder.pkl'))
    
    if model_name == "ANN":
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'{place_name}_best_model.h5'))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, f'{place_name}_best_model.pkl'))
        
    return scaler, le, model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard/<place_name>')
def dashboard(place_name):
    comparison_data, best_model_name, best_model_reason = get_model_info(place_name)
    feature_importance_exists = os.path.exists(os.path.join(STATIC_DIR, f'{place_name}_feature_importance.png'))
    return render_template('dashboard.html', 
                           place_name=place_name, 
                           comparison_data=comparison_data, 
                           best_model_name=best_model_name,
                           best_model_reason=best_model_reason,
                           feature_importance_exists=feature_importance_exists,
                           explanations=EXPLANATIONS)

@app.route('/predict/<place_name>', methods=['GET', 'POST'])
def predict(place_name):
    comparison_data, best_model_name, _ = get_model_info(place_name)
    prediction = None
    angle = None
    
    if request.method == 'POST':
        # Collect features from form
        features = ['Temp', 'Humidity', 'SoilPercent', 'RawSoil', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Angle']
        input_data = [float(request.form.get(f)) for f in features]
        angle = int(request.form.get('Angle'))
        
        # Load models and process
        scaler, le, model = load_predictor(place_name, best_model_name)
        
        # Scale input (using same scaler)
        input_df = pd.DataFrame([input_data], columns=features)
        X_scaled = scaler.transform(input_df)
        
        # Predict
        if best_model_name == "ANN":
            pred_probs = model.predict(X_scaled)
            pred_idx = np.argmax(pred_probs, axis=1)[0]
        else:
            pred_idx = model.predict(X_scaled)[0]
            
        prediction = le.inverse_transform([pred_idx])[0]
        
    return render_template('predict.html', 
                          place_name=place_name, 
                          prediction=prediction, 
                          angle=angle,
                          best_model_name=best_model_name)

# Vercel needs the 'app' variable to be exposed
