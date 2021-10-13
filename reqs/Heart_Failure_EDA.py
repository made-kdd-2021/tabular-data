#!/usr/bin/env python
# coding: utf-8

# ## Курсовой проект по курсу "Интерпретируемый ИИ и майнинг данных"
# ### Часть 1: Работа с табличными данными
# ### Этап 1: EDA and Preprocessing

# **Материалы к проекту (файлы):**
# heart.csv
# 
# **Целевая переменная:**
# HeartDisease: output class [1: heart disease, 0: Normal]
# 
# **Описание датасета:**
# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
# 
# **Атрибуты:**
# 1. Age: age of the patient [years] - **Возраст**
# 2. Sex: sex of the patient [M: Male, F: Female] - **Пол**
# 3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic] - **Тип боли в груди**
# 4. RestingBP: resting blood pressure [mm Hg] - **Артериальное давление в покое**
# 5. Cholesterol: serum cholesterol [mm/dl] - **Холестерин** 
# 6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] - **Уровень сахара в крови натощак**
# 7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria] - **ЭКГ в покое**
# 8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] - **Максимальная частота сердечных сокращений**
# 9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No] - **Стенокардия, вызванная физической нагрузкой**
# 10. Oldpeak: oldpeak = ST [Numeric value measured in depression] - **ST в покое**
# 11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping] - **ST при пиковой нагрузке**
# 12. HeartDisease: output class [1: heart disease, 0: Normal] - **Сердечный приступ**

# ## Шаг 1: Подготовка инструментов

# ### 1.1 Необходимые модули

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.2 Загрузка данных

# #### Загрузим данные и посмотрим основные статистики

# In[2]:


DATASET_PATH = 'data/heart.csv'


# In[3]:


df_base = pd.read_csv(DATASET_PATH)
df = df_base.copy()
df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# #### Выводы:
# * 918 объектов
# * 11 признаков и 1 целевая переменная
# * Пропусков в данных нет
# * Дубликатов в данных нет
# * Признаки разного типа
# 

# ## Шаг 2: Анализ данных

# ### 2.1 Целевой признак

# In[8]:


df['HeartDisease'].value_counts()


# In[9]:


df['HeartDisease'].hist();


# Есть небольшой дисбаланс - некритичный

# In[10]:


disbalance = df['HeartDisease'].value_counts()[1] / df['HeartDisease'].value_counts()[0]
disbalance


# ### 2.2 Анализ признаков

# In[11]:


num_features = df.select_dtypes(include=[np.number]).columns.to_list()
cat_features = df.select_dtypes(include=[np.object]).columns.to_list()
print(f'Числовые признаки({len(num_features)}): {num_features}')
print(f'Категориальные признаки({len(cat_features)}): {cat_features}')


# #### 2.2.1 Числовые признаки

# In[12]:


fig = plt.figure(figsize=(6,4)) 
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap='Blues', annot=True, mask=mask, linewidth=0.5)
plt.show()


# In[13]:


color ='#8abbd0'

for i in num_features:
    fig, ax = plt.subplots(1,4, figsize=(15,3))
    fig.patch.set_facecolor( '#F2F2F2')
    sns.histplot(df[i], bins=10, ax=ax[0],  color=color, kde=True)
    ax[0].lines[0].set_color('#F97A1F')
    sns.kdeplot(x=i,data=df, hue='HeartDisease',ax=ax[1],shade=True, alpha=0.3)
    sns.boxplot(x=i, data=df,ax=ax[2], color=color)
    sns.boxplot(x=i, data=df, hue='HeartDisease',y=[""]*len(df),ax=ax[3],palette=['#8abbd0','#F97A1F'],boxprops=dict(alpha=.3))
    plt.tight_layout


# In[14]:


sns.pairplot(df[num_features], hue='HeartDisease');


# #### Выводы:
# Наиболее высокий шанс получить сердечный приступ:
# * Age > 55
# * FastingBS == Yes
# * MaxHR < 150
# * Oldpeak > 1
# 
# RestingBP имеет экстремальные выбросы в нуле.<br>
# Cholesterol имеет много нулевых значений.<br>
# Корреляция между признаками в целом невысокая.

# #### 2.2.2 Категориальные признаки

# In[15]:


print(len(cat_features), 'категориальных признаков:', "\n", cat_features, '\n')
for i in cat_features:
    unique_no = df[i].nunique()
    unique_name = df[i].unique().tolist()
    print('Признак', i, 'имеет', unique_no, 'уникальных значения:')
    print(unique_name, "\n")


# In[16]:


palette = ['#8abbd0', '#FB9851', '#36E2BD','#D0E1E1']

for feature in cat_features:
    fig, ax = plt.subplots(1,3, figsize=(15,3))
    fig.patch.set_facecolor('#F2F2F2')

    sns.countplot(x=df[feature], data=df, ax=ax[0], palette=palette, alpha=0.8)
    for p, label in zip(ax[0].patches, df[feature].value_counts().index):
        ax[0].annotate(p.get_height(), (p.get_x()+p.get_width()/3, p.get_height()*1.03))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
                    
    df[feature].value_counts().plot.pie(autopct='%1.1f%%', startangle = 90, ax=ax[1], colors=palette, frame=True)
    ax[1].set_ylabel('')
    ax[1].set_title(feature)

    sns.histplot(x=feature,data=df, hue='HeartDisease',ax=ax[2], alpha=0.3, shrink=.8)  
    
    plt.tight_layout


# #### Выводы:
# Наиболее высокий шанс получить сердечный приступ:
# * Sex == M
# * ChestPainType == ASY
# * ExerciseAngina == Y
# * ST_Slope == Flat

# ## Шаг 3: Подготовка данных

# ### 3.1 Обработка выбросов

# In[17]:


df.loc[df['RestingBP']==0]


# Выброс всего один, поэтому просто удалим его.

# In[18]:


row = df[df['RestingBP']==0].index
df = df.drop(df.index[row])


# ### 3.2 Обработка нулевых значений

# In[19]:


df.loc[df['Cholesterol']==0]


# Холестерин не может быть равен нулю, поэтому заменим его медианным значением.

# In[20]:


median_values = df['Cholesterol'].median()
row = df[df['Cholesterol']==0].index
df.loc[row, 'Cholesterol'] = median_values


# ### 3.3 Кодирование категориальных признаков

# Всем категориальным признакам поставим в соответствие наборы из 0 и 1:

# In[21]:


df.head(3)


# In[22]:


target_name = 'HeartDisease'
target = df[target_name]
df = df.drop([target_name], axis=1)
df.head(3)


# In[23]:


for feature in cat_features:
    df[feature] = df[feature].astype(object)
df_encoded = pd.get_dummies(df)
df_encoded.head(3)


# ### 3.4 Масштабирование числовых признаков

# In[24]:


num_columns = num_features[:-1]
scaler = MinMaxScaler()
scaler.fit(df_encoded)
df_scaled = scaler.transform(df_encoded)
df_scaled


# In[25]:


df_prepared = pd.DataFrame(data=df_scaled, index=df_encoded.index, columns=df_encoded.columns)


# In[26]:


df_prepared.head()


# ### 3.5 Разобьем данные на обучающую, тестовую и валидационную выборки:

# In[27]:


(X, X_valid, y, y_valid) = train_test_split(df_prepared, target, test_size=0.1, random_state=0, stratify=target)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# In[28]:


print(f'Обучающая выборка: X_train:{X_train.shape}/y_train:{y_train.shape}')
print(f'Тестовая выборка: X_test:{X_test.shape}/y_test:{y_test.shape}')
print(f'Валидационная выборка: X_valid:{X_valid.shape}/y_valid:{y_valid.shape}')


# ### 3.6 Сохраним полученные выборки

# In[29]:


X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
X_valid.to_csv('data/X_valid.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
y_valid.to_csv('data/y_valid.csv', index=False)

