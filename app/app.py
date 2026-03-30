from flask import Flask, request, render_template
from src.models.predict import predict_baseline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        prediction = predict_baseline(text)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)