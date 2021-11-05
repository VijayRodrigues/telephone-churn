import pandas as pd
import numpy as np
import pickle

tel_df = pd.read_csv('Telecom_customer_churn.csv')
tel_df.drop(['TotalCharges', 'customerID'], axis = 1, inplace=True)

from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()

tel_df["gender"] = lab_enc.fit_transform(tel_df["gender"])
tel_df["Partner"] = lab_enc.fit_transform(tel_df["Partner"])
tel_df["Dependents"] = lab_enc.fit_transform(tel_df["Dependents"])
tel_df["PhoneService"] = lab_enc.fit_transform(tel_df["PhoneService"])
tel_df["MultipleLines"] = lab_enc.fit_transform(tel_df["MultipleLines"])
tel_df["InternetService"] = lab_enc.fit_transform(tel_df["InternetService"])
tel_df["OnlineSecurity"] = lab_enc.fit_transform(tel_df["OnlineSecurity"])
tel_df["OnlineBackup"] = lab_enc.fit_transform(tel_df["OnlineBackup"])
tel_df["DeviceProtection"] = lab_enc.fit_transform(tel_df["DeviceProtection"])
tel_df["TechSupport"] = lab_enc.fit_transform(tel_df["TechSupport"])
tel_df["StreamingTV"] = lab_enc.fit_transform(tel_df["StreamingTV"])
tel_df["StreamingMovies"] = lab_enc.fit_transform(tel_df["StreamingMovies"])
tel_df["Contract"] = lab_enc.fit_transform(tel_df["Contract"])
tel_df["PaperlessBilling"] = lab_enc.fit_transform(tel_df["PaperlessBilling"])
tel_df["PaymentMethod"] = lab_enc.fit_transform(tel_df["PaymentMethod"])
tel_df["Churn"] = lab_enc.fit_transform(tel_df["Churn"])


X = tel_df.drop(['Churn'], axis = 1)
y = tel_df['Churn']


from imblearn.over_sampling import SMOTE
SM = SMOTE()
x_over, y_over = SM.fit_resample(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.30, random_state = 200)


from sklearn.ensemble import GradientBoostingClassifier
mod_grad_class = GradientBoostingClassifier(tol= 0.01, random_state= 150, n_estimators= 150, 
                                            max_features= 'auto', loss= 'deviance', criterion= 'friedman_mse')

mod_grad_class.fit(X_train,y_train)


pickle.dump(mod_grad_class , open('TelChurn.pkl', 'wb'))