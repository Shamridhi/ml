import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("data.csv")


print(df.head())


X = df[["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphotase", 
        "Alamine Aminotransferase", "Aspartate Aminotransferase", "Total Protiens", 
        "Albumin", "Albumin_and_Globulin_Ratio"]]
y = df["Result"]


label_encoder = LabelEncoder()
X["Gender"] = label_encoder.fit_transform(X["Gender"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


with open("model.pkl", "wb") as file:
    pickle.dump(classifier, file)
