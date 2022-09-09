import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


book = r"D:\condamini\MachineLearningTutorail\Tutorial1\missdata\Book1.csv"
data = pd.read_csv(book, header=None)

print(data)

x = data.values
imp = Imputer(missing_value=np.nan, strategy='mean')
imp.fit(x)
result = imp.transform(x)
print(result)
