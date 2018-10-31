
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/katari/Desktop/git/Medical_cost_dataset/src/data/insurance.csv')

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x = 'region', data = df, palette = "PiYG", orient = 'h', ax = ax, edgecolor = '1')
for i in ax.patches:
    ax.text(i.get_x()+0.3,i.get_height()+3,str(round((i.get_height()/df.region.shape[0])*100))+'%',           fontsize = 12)
ax.set_xlabel("Region", fontsize = 13)
ax.set_title("Region Distribution", fontsize = 15)
ax.tick_params(length=5,labelsize=12,labelcolor = 'black')
x_axis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left= True)
plt.show()
print("minimum age:",df['age'].min())
print("maximum age:",df['age'].max())
# classify age into 3 groups
# Young adult(18-25), Adult(26-50), Senior(51-64)
# convert continious variable 'age' to categorical variable  

cut_points = [17,25,50,64]
labels = ['Young adult', 'Adult', 'Senior']
df['age_category'] = pd.cut(df["age"], cut_points, labels =  labels)
set(list(df['age_category']))

#Age distribution by categories

f, (ax,ax2) = plt.subplots(2,1,figsize = (8,10))
sns.countplot(x = 'age_category',data = df, palette = 'Pastel2',orient = 'v',ax = ax, edgecolor = '1')
for i in ax.patches:
    ax.text(i.get_x()+0.3,i.get_height()+3,           str(round((i.get_height()/df.age_category.shape[0])*100))+'%',fontsize =12)
ax.set_xlabel("Age Categories",fontsize =13)
ax.tick_params(length=5, labelsize = 12, labelcolor = 'black')
ax.set_title("Age Distribution by Categories",fontsize =15)

ax2.hist('age',bins = 10,data = df, edgecolor = '0.1')
ax2.set_xlabel("Age",fontsize =13)
ax2.tick_params(length=5, labelsize = 12, labelcolor = 'black')
ax2.set_title("Age Distribution",fontsize =15)
x_axis = ax.axes.get_yaxis().set_visible(False)
f.subplots_adjust(hspace = 0.5)
sns.despine(left=True)
plt.show()

def gender_dist_plot(x_axis,title):
    f,ax = plt.subplots(figsize=(10,5))
    sns.countplot(x=x_axis, data = df, ax = ax,palette=['dodgerblue','lightpink']
                  ,hue='sex', hue_order=['male','female'] )

    for i in ax.patches:
        ax.text(i.get_x()+0.1, i.get_height()+3,                str(round((i.get_height()/df.region.shape[0])*100))+'%')
    ax.set_title(title+ ' Distribution by Gender', fontsize = 15)
    ax.set_xlabel(title, fontsize =12)
    ax.tick_params(length=5, labelsize= 12, labelcolor = 'black')
    x_axis = ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=[1,0.8],fontsize = 12, title = "Gender Type",ncol=2)
    sns.despine(left = True)
    plt.show()

gender_dist_plot("age_category",'Age Category')
gender_dist_plot("region",'Region')
f, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='region', data = df, ax = ax , hue = "smoker", palette=["C7", "C9"])
for i in ax.patches:
    ax.text(i.get_x()+0.1, i.get_height()+3,
               str(round((i.get_height()/df.region.shape[0])*100))+'%')
ax.set_xlabel("Region",fontsize=13)
ax.set_title("Regional Distribution of Smokers",fontsize =15)
ax.tick_params(length =5, labelsize=12)
xaxis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left = True)
plt.show()

from scipy import stats
from scipy.stats import norm, skew, kurtosis

def data_transform(data,input):
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,8))
    
    sns.boxplot(x =input, data= data, ax = ax1, orient='v')
    sns.distplot(data[input],ax = ax2, color = 'blue', hist = False)
    res = stats.probplot(data[input], plot = ax3)
    
    axes = [ax1,ax2]
    kwargs = {'fontsize':14,'color':'black'}
    ax1.set_title(input+' Boxplot Analysis',**kwargs)
    ax1.set_xlabel('Box',**kwargs)
    ax1.set_ylabel('BMI Values',**kwargs)

    ax2.set_title(input+' Distribution',**kwargs)
    ax2.set_xlabel(input+' values',**kwargs)

    ax3.set_title('Probability Plot',**kwargs)
    ax3.set_xlabel('Theoretical Quantiles',**kwargs)
    ax3.set_ylabel('Ordered Values',**kwargs)
    f.subplots_adjust(wspace=0.22,right= 2)
    sns.despine()
    
    return plt.show()

    

data_transform(df,'bmi')


# ### Categorize BMI value

# Underweight if bmi value is between 14 - 18.99
# Normal if bmi value is btw 19 - 24.99
# Overweight if bmi value is btw 25 - 29.99
# Obese if bmi value is above 30


cut_points = [14,19,25,30,65]
label_names = ['Underweight',"normal","overweight","obese"]
df["bmi_cat"] = pd.cut(df['bmi'],cut_points,labels=label_names)
gender_dist_plot('bmi_cat','BMI')


# ### Charges feature analysis

data_transform(df,'charges')

The Charges feature is not normally distributed.
The feature is affected by outliers
The distribution left skewedTo solve this problems natural Log transformation is performed on the Charges feature

df.charges = np.log1p(df.charges)
data_transform(df,'charges')


# ### Scatter Plot Analysis
sns.lmplot(x = "bmi", y= "charges", hue = "smoker",data = df, 
           size = 6, aspect = 1.3,
          scatter_kws={"s": 50, "alpha": 1,'edgecolor':'black'}
          ,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.xlabel('BMI',fontsize=12)
plt.ylabel('Charge',fontsize=12)
plt.show()

# From the above Scatter plot

# 1. The charges for smoker are higher than for non somkers in general
# 2. For smokers, there is a linear relationship between BMI and charges.    With increase in BMI value Charges also increase.
# 3. For Non-smokers, charges does not depend on the BMI value



plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df.corr(),annot=True,robust=True)
plt.title('Correlation Analysis on the Dataset',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()

# From the above Heatmap there is a strong correlation between age 
# and charges. But from the correlation plot it is observed that there is 
# correlation between age and bmi for smokers

df.drop(['age_category','bmi_cat'],axis=1,inplace=True)
df_smoker = df[df.smoker=='yes']
df_smoker.head()
df_smoker.to_csv('/Users/katari/Desktop/git/Medical_cost_dataset/src/data/smoker_data.csv')

