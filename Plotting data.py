import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sbn

# import heart failure dataset
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# check for missing data
dataset.isnull().sum()

plt.rcParams['figure.figsize'] = 15, 6

x = dataset.iloc[:, :-1]  # input variables
y = dataset.iloc[:, -1]  # output variable

# feature importance plot
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_, index=x.columns)
feat_imp.nlargest(12).plot(kind='barh')
plt.show()

# boxplot of variable (e.g. serum_creatinine)
sbn.boxplot(x=dataset.serum_creatinine)
plt.show()
