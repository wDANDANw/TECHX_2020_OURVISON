from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/haha")
def haha():
    return render_template("haha.html")

if __name__ == "__main__":
    app.run(debug=True)