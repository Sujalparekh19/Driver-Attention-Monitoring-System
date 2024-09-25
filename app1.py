from flask import Flask, redirect, url_for, render_template, request
import os
from index import d_dtcn

secret_key = str(os.urandom(24))

app = Flask(__name__)
app.config['TESTING'] = True
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key

# Defining the home page of our site
@app.route("/", methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
            return render_template("test1.html")
    else:
        return render_template("index.html")

@app.route("/start", methods=['GET', 'POST'])
def index():
    print("Request method:", request.method)
    if request.method == 'POST':
        if request.form.get('Start') == 'Start':
            print("Start button clicked. Initiating d_dtcn.")
            shape_predictor_path = "C:\\Users\\sujal\\.vscode\\Drowsiness-Detection-master\\Drowsiness-Detection-master\\shape_predictor_68_face_landmarks.dat"
            d_dtcn(shape_predictor_path)
            print("d_dtcn completed.")
            return render_template("index.html")
    else:
        print("GET request received. Rendering index.html.")
        return render_template("index.html")

@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contact.html')

if __name__ == "__main__":
    app.run()
