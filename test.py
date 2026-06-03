import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("domain_dataset up.csv")

# Features and Target
X = df.drop("Domain", axis=1)
y = df["Domain"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Sample User
user = [[
    1,0,0,0,0,
    1,1,1,1,
    0,0,0,0,
    1,1
]]

# Get Probabilities
probabilities = model.predict_proba(user)[0]
domains = model.classes_

results = list(zip(domains, probabilities))
results.sort(key=lambda x: x[1], reverse=True)

# Output
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nTop Recommendations:")

for i, (domain, prob) in enumerate(results[:3], start=1):
    print(f"{i}. {domain} ({prob * 100:.2f}%)")

print("=" * 50)