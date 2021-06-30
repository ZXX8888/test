import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
df = pd.read_csv("train.csv")
df.drop_duplicates()
strs = df['target'].value_counts()
value_map = dict((v, i) for i,v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5':4,'Class_6':5,'Class_7':6,'Class_8':7,'Class_9':8}
df = df.replace({'target':value_map})
df = df.drop(columns=['id'])
x_train = df.iloc[:, :-1]
y_train = df['target']
SMOTE().fit_resample(x_train, y_train)
LDA(n_components=8).fit_transform(x_train,y_train)
df = pd.read_csv("test.csv")
x_test = df.iloc[:, 1:] 
model = CascadeForestClassifier(n_jobs=-1,use_predictor=True, n_estimators=3, n_trees=300,max_depth=12,
                                min_samples_leaf=2)
model.fit(x_train.values, y_train.values)
y_pred = model.predict(x_test.values)
proba = model.predict_proba(x_test.values)
output = pd.DataFrame({'id': x_test.index, 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('my_submission15.csv', index=False)
df = pd.read_csv("F:\\homework\\机器学习\\竞赛大作业\\test.csv")
output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('my_submission15.csv', index=False)