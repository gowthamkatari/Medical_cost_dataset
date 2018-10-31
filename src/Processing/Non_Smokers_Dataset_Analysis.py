
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_non_smoker = pd.read_csv('/Users/katari/Desktop/git/Medical_cost_dataset/src/data/Non_smoker_data.csv')

sns.lmplot(x = 'bmi',y='charges',hue=None,data=df_non_smoker,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.xlabel('BMI',fontsize=12)
plt.ylabel('Charge',fontsize=12)
plt.show()

# Convert all categorical columns in the dataset to Numerical for the Analysis
df_non_smoker['children'] = df_non_smoker['children'].astype('category')
df_non_smoker = pd.get_dummies(df_non_smoker,drop_first=True)
#correlation Analysis
plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_non_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()


# only age has strong correlation with charges

# Let plot the age vs. charge scatter plot to see the correlation between them
sns.lmplot(x = 'age',y='charges',data=df_non_smoker,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.show()

