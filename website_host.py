import os

import turicreate as tc
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.abspath(os.getcwd()) + "/uploads"


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/uploads', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template("home.html", error="Unable to load your image! Please try again")
        filename = secure_filename(f.filename)
        predict_filepath = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        f.save(predict_filepath)
        return redirect(url_for('predict', predict_file_name=filename))


@app.route('/predict?imageName=<predict_file_name>')
def predict(predict_file_name):
    test_image = tc.SFrame({'image': [tc.Image("uploads/" + predict_file_name)]})
    print(test_image)
    model = tc.load_model('digits.model')
    test_image['prediction'] = model.predict(test_image)
    print("The prediction is " + test_image['prediction'])
    return render_template("predict.html", prediction=test_image['prediction'][0])


@app.route('/back_to_home_page')
def back_to_home_page():
    return render_template("home.html")


app.run(host="localhost")
