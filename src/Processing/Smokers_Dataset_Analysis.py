
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_smoker = pd.read_csv('/Users/katari/Desktop/git/Medical_cost_dataset/src/data/smoker_data.csv')

#converts categorical to nummerical values
df_smoker = pd.get_dummies(df_smoker,drop_first=True)
df_smoker.head()

plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()

df_smoker.drop(['children','sex_male', 'region_northwest',
       'region_southeast', 'region_southwest'],axis=1,inplace=True)

sns.lmplot(x = 'bmi',y='charges',hue=None,data=df_smoker,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.xlabel('BMI',fontsize=12)
plt.ylabel('Charge',fontsize=12)
plt.show()

