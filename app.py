import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_feature = [float(i) for i in request.form.values()]
    feature_value = [np.array(input_feature)]
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                     'mean smoothness', 'mean compactness', 'mean concavity',
                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                     'radius error', 'texture error', 'perimeter error', 'area error',
                     'smoothness error', 'compactness error', 'concavity error',
                     'concave points error', 'symmetry error', 'fractal dimension error',
                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                     'worst smoothness', 'worst compactness', 'worst concavity',
                     'worst concave points', 'worst symmetry', 'worst fractal dimension']

    df = pd.DataFrame(feature_value, columns=feature_names)
    output = model.predict(df)
    if output == 0:
        res_val = 'breast cancer'
    else:
        res_val = 'no breast cancer'
    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == '__main__':
    app.run()