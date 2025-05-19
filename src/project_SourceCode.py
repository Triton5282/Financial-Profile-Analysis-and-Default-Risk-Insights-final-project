from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv("D:/CSE/Financial-Profile-Analysis-and-Default-Risk-Insights-final-project/Dataset/LoanData_Raw_v1.0.csv")
print("before conversion:\n",df.info())

df['default'] = pd.to_numeric(df['default'], errors='coerce').fillna(0).astype('int')
print("after conversion:\n",df.info())

print("shape of the dataset:\n",df.shape)

print("before handalling null count:\n")
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

df['age'] = df['age'].fillna(df['age'].mean())
df['income'] = df['income'].fillna(df['income'].mean())
df['ed'] = df['ed'].fillna(df['ed'].mode()[0])


print("after handalling null count:\n")
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])



# Feature and target split
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LinearSVC Model
linear_model = LinearSVC(penalty='l1')
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
print("LinearSVC Accuracy:", accuracy_score(y_test, y_pred_linear))
print("\nClassification Report:\n", classification_report(y_test, y_pred_linear))


# RBF SVC Model
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_model.predict(X_test_scaled)
print("RBF SVC Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rbf))