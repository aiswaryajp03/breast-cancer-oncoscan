import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import roc_auc_score, classification_report, confusion_matrix


# --- 1. Load & Clean Data ---
df = pd.read_csv("Breast_Cancer.csv")
df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore', inplace=True)

print("Dataset shape (rows, columns):", df.shape)
print("\nMalignant vs Benign counts:")
print(df['diagnosis'].value_counts())


# --- 2. Encode Target (M=1, B=0) ---
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# --- 3. Train / Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# --- 4. Feature Selection via Correlation Matrix ---
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]

print(f"\nDropping {len(to_drop)} highly correlated features:")
print(" ", to_drop)

X_train = X_train.drop(columns=to_drop)
X_test  = X_test.drop(columns=to_drop)

print(f"\nRemaining features ({len(X_train.columns)}):")
print(" ", list(X_train.columns))


# --- 5. Model Comparison ---
models = {
    "Logistic Regression": Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=5000))
    ]),
    "SVM": Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ]),
    "Random Forest": Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-Validation ROC-AUC Scores:")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")


# --- 6. Final Model: SVM (best performer) ---
print("\nSelected Model: SVM")
final_pipeline = models["SVM"]
final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]

print(f"\nTest AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --- 7. Save Model Assets ---
joblib.dump(final_pipeline.named_steps['classifier'], 'model.pkl')
joblib.dump(final_pipeline.named_steps['scaler'],     'scaler.pkl')
joblib.dump(to_drop,                                  'removed_columns.pkl')
joblib.dump(list(X_train.columns),                    'feature_columns.pkl')

print("\nSaved: model.pkl, scaler.pkl, removed_columns.pkl, feature_columns.pkl")
