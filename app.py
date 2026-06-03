from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("domain_dataset up.csv")

# Features and Target
X = df.drop("Domain", axis=1)
y = df["Domain"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    python = 1 if "python" in request.form else 0
    sql = 1 if "sql" in request.form else 0
    ml = 1 if "ml" in request.form else 0
    deeplearning = 1 if "deeplearning" in request.form else 0
    statistics = 1 if "statistics" in request.form else 0
    linux = 1 if "linux" in request.form else 0
    networking = 1 if "networking" in request.form else 0
    docker = 1 if "docker" in request.form else 0
    git = 1 if "git" in request.form else 0
    html = 1 if "html" in request.form else 0
    css = 1 if "css" in request.form else 0
    javascript = 1 if "javascript" in request.form else 0
    react = 1 if "react" in request.form else 0
    cloud = 1 if "cloud" in request.form else 0
    aws = 1 if "aws" in request.form else 0

    user = [[
        python,
        sql,
        ml,
        deeplearning,
        statistics,
        linux,
        networking,
        docker,
        git,
        html,
        css,
        javascript,
        react,
        cloud,
        aws
    ]]

    # Get probabilities
    probabilities = model.predict_proba(user)[0]

    # Get domain names
    domains = model.classes_

    # Combine and sort
    results = list(zip(domains, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)
    # Top 2 domains
    primary_domain = results[0][0].replace("_", " ")
    secondary_domain = results[1][0].replace("_", " ")

    print("User Input:", user)

    print("Results:", results)

    return render_template(
        "result.html",
        primary=primary_domain,
        secondary=secondary_domain
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)