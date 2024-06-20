import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("block_data.csv")
print(df)

%matplotlib inline
plt.xlabel('timestamp(seconds)')
plt.ylabel('blockreward(ether)')
plt.scatter(df.timestamp,df.blockReward,color="blue",marker=".")

model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df.timestamp.values.reshape(-1,1),df.blockReward.values.ravel())

new_timestamp=np.array([[1686640499]])
predicted_blockReward=model.predict(new_timestamp)
plt.scatter(df.timestamp,df.blockReward,label="data points")
plt.scatter(new_timestamp,predicted_blockReward,color="red",label="predicted block reward")
plt.xlabel('timestamp')
plt.ylabel('blockreward(ether)')
plt.legend()
plt.show()

print("predicted blockreword", predicted_blockReward[0])
