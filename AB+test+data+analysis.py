
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt


# In[2]:

directory = "/Users/abittaraev/Desktop/test_task_mail_ru/AB_Test_Results.xlsx"
df = pd.read_excel(directory, header=0)


# In[3]:

df.sort_values("USER_ID", inplace = True)
df.reset_index(inplace=True, drop=True)
df.info() #check for missing data
df.head(10)


# Видим, что для некоторых юзеров были применены как экспериментальные условия так и контрольные. Также для некоторых юзеров проводились репликации эксперимента. У нас нет оснований предполагать, что в дизайн эксперимента был заложен план кроссовер, так как нет временных отметок. Поэтому в рамках простого аб-теста безопаснее удалить юзеров, для которых есть обе отметки (и вариант и контроль), а репликации суммировать.

# In[6]:

# count treatments for user_id
df_gr=df.groupby('USER_ID')['VARIANT_NAME'].nunique().reset_index()
df_gr.head(10)


# In[7]:

# find re-used user_ids and remove them
df_to_del = df_gr.loc[df_gr.VARIANT_NAME > 1, ['USER_ID']]
df_mg = pd.merge(df, df_to_del, how='left', on='USER_ID', indicator=True)
df_corr = df_mg.loc[df_mg._merge == 'left_only', ['USER_ID', 'VARIANT_NAME', 'REVENUE']]
df_corr.head(10)


# In[8]:

# sum replicated revenues per user_id
df_sum = df_corr.groupby(['USER_ID', 'VARIANT_NAME'])['REVENUE'].sum().reset_index()
df_sum.head(10)


# In[9]:

# summary stats
df_sum.groupby("VARIANT_NAME")["REVENUE"].describe()


# Распределения переменной REVENUE (даже если суммировать реплицированные значения по каждому юзеру) определенно ассиметричны. 
# Тот факт, что $ \gt 75\% $ наблюдений в обеих группах - это нулевые значения говорит, что наиболее подходящей моделью обоих распределений была бы модель  ZIPR (zero inflated poisson regression). 
# В рамках этого эксперимента мы выберем модель попроще. Мы будем считать, что  распределение Пуассона лежит в основе дата-генерационных процессов: $ X_{test} \sim Poisson(\lambda_1) $  и $ X_{control} \sim Poisson(\lambda_2)$ 
# 

# In[10]:

#recode REVENUE as binary variable (convert / non-convert) to represent Poisson counts per time unit
df_sum['convert'] = df_sum.REVENUE.apply(lambda x: 1 if x > 0.00  else 0)
df_sum.groupby("VARIANT_NAME")["convert"].describe()


# Судя по этим  статистикам, разница в интенсивности конверсии по группам будет незначимой. Следующим шагом, проведем _rateratio_ тест.

# Гипотезы _rateratio_ test: 
# * $ H_0: \theta = \frac{\lambda_1}{\lambda_2} \leq 1 $
# * $ H_a: \theta  = \frac{\lambda_1}{\lambda_2} \gt 1 $
#      
# В Python нет готового _rateratio_ теста похожего на R 
# [link to CRAN](https://cran.r-project.org/web/packages/rateratio.test/vignettes/rateratio.test.pdf), 
# но мы можем  использовать exact binomial test (scipy.stats.binom_test), опираясь на следующие свойства распределения Пуассона: 
# * Если $ X_{test} \sim {Poisson(\lambda_1)} $ и  $ X_{control} \sim {Poisson(\lambda_2)} $ независимы, то 
# $ X_{test}  \lvert  X_{test} + X_{control} =  k \sim {Binomial(k,p(\theta))} $, где k=количество испытаний и 
# $p(\theta) = \frac{\lambda_1}{\lambda_1 + \lambda_2}$.
# * $X_{1},X_{2},...,X_{n_1}$ are iid $Poisson(\lambda)$, тогда $X = \sum_{i=1}^{n} X_{i} \sim Poisson(n\lambda)$
# [link to article](https://userweb.ucs.louisiana.edu/~kxk4695/JSPI-04.pdf)
# 
# Чтобы задать параметры биномиального теста проделаем следующую процедуру:
# 
# * Перепишем параметры $\lambda_1,\lambda_2$ как $ r_1 = n\lambda_1, r_2 = m\lambda_2$ согласно свойству выше. Параметры n и m - это размеры выборок из двух сравниваемых популяций.
# * Можем записать  $p(\theta)=\frac{\lambda_1}{\lambda_1 + \lambda_2}$ как $\frac{r_1}{r_1 + r_2}$ => $\frac{n\lambda_1}{n\lambda_1 + m\lambda_2} $
# * $\lambda_1= \theta\lambda_2 $, следовательно $\frac{n\lambda_1}{n\lambda_1 + m\lambda_2} 
#     = \frac{n\theta\lambda_2}{n\theta\lambda_2 + m\lambda_2} = \frac{n\theta}{n\theta + m}$
# *  Таким образом, если $H_0$ верна и $\theta \leq 1$, то тестируем $H_0: \pi = p(\theta)\leq \frac{n}{n+m}$ против $H_a: \pi = p(\theta) \gt \frac{n}{n+m}$, где $\pi$ = обычная пропорция (= вероятность) успехов в биномиальном распределении

# In[11]:

#compute sample sizes and pi
dff=df_sum.groupby("VARIANT_NAME")["USER_ID"].count().reset_index()
n = dff.loc[dff["VARIANT_NAME"]=='variant']["USER_ID"].values[0]
m = dff.loc[dff["VARIANT_NAME"]=='control']["USER_ID"].values[0]
pi=n/(n+m)
print("Sample size test: % 2d" %(n) )  
print("Sample size control : % 2d" %(m) )  
print("Proportion of succeses under H_0 : % 5f" %(pi) )  



# In[12]:

#successes in the test group
dff_s=df_sum.groupby("VARIANT_NAME")["convert"].sum().reset_index()
s1= dff_s.loc[dff["VARIANT_NAME"]=='variant']["convert"].values[0]
s2= dff_s.loc[dff["VARIANT_NAME"]=='control']["convert"].values[0]
print("Successes test: % 2d" %(s1))  
print("Successes control: % 2d" %(s2))  


# In[14]:

p_val=scipy.stats.binom_test(s1,s1+s2,pi,alternative="greater")
print("P_value of the test : % 5f" %(p_val))  
print("Cannot reject H_0")


# ##### Выводы
# * Разница в конверсиях пользователей статистически не значима, то есть тест не подтверждает  эффективность эксперимента
# * Строго говоря эксперимент сам по себе некорректен, так как для чистого эксперимента один юзер может появляться в выборке только один раз и только в одной группе. Строго говоря, эксперимент нужно переделать, принимая эти ограничения во внимание
# * Для правильного анализа эксперимента (при наличии большого количества значений переменной revenue=0), нужно использовать более сложные методы, такие как ZIPR
