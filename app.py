from flask import Flask
from markupsafe import escape
from flask import render_template
import flask
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    model = pickle.load(open('model/model_RF.pkl','rb'))
    prediction = model.predict(final_features)
    output = {1:'snow',0:'rain'}
    return flask.render_template('main.html',prediction_text='Prediction of Precip Type: {}'.format(output[prediction[0]]))


@app.route('/')
def main():
    return(render_template('main.html'))
if __name__ == '__main__':
    app.run(debug=True)
