import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/placement.csv")

# clean spaces
df.columns = df.columns.str.strip()


# convert to numeric
df["placement_status"] = df["placement_status"].map({
    "Placed": 1,
    "Not Placed": 0
})
df=df.drop(['student_id','salary_package_lpa'],axis=1)
df=pd.get_dummies(df,drop_first=True)
X=df.drop('placement_status',axis=1)
y=df['placement_status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print("Accuracy:", accuracy)
print("Accuracy:", accuracy)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\n Feature Importance:\n")
print(importance.head(10))
