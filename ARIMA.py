import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('../total.csv')
df = df.loc[:10316, :]
df = df.drop(columns=['DATE'])
train = df.loc[:8252, :]
test = df.loc[8252:, :]

q = range(0, 2)
d = range(0, 2)
p = range(0, 5)
pdq = list(itertools.product(p, d, q))

if __name__ == '__main__':
    out = []
    for i in range(len(train.columns)):
        AIC = []
        models = []
        index_train = train.iloc[:, i]
        index_test = test.iloc[:, i]

        for param in pdq:
            try:
                mod = ARIMA(index_train, param)
                results = mod.fit()
                print('ARIMA{} - AIC:{}'.format(param, results.aic), end='\r')
                AIC.append(results.aic)
                models.append(param)
            except:
                continue

        print('The smallest AIC is {} for model ARIMA{}'.format(min(AIC), models[AIC.index(min(AIC))]))
        mod = ARIMA(index_train, models[AIC.index(min(AIC))])
        result = mod.fit()
        out.append(result.forecast()[0])
        print(out)
        csv = pd.Series(out)
        csv.to_csv('../output.csv')