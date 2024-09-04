import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming you have your original training data
x_train = np.load('x_train.npy')

# Initialize and fit the scaler
sc = StandardScaler()
sc.fit(x_train)

# Save the scaler
joblib.dump(sc, 'scaler.pkl')
