---
title: Linear Regression and Simulation Methods Final Exam
author:
  name: Pierre Lague & Mattéo Chopin
  link: 
date: 2022-05-21 09:45:00 +0800
categories: [Studies, UBS - L3 CMI, Statistics]
tags: [Python, Exam, Regression, English]
math: true
mermaid: true
image:
  src: '/assets/posts/regression_lineaire_exam/header.jpg'
  width: 800
  height: 600
---

Final exam on computers for the **Linear Regression and Simulation Methods** module taught by Pr. François Septier (U. of South Brittany).

# Linear Regression and simulation methods exam. {#linear-regression-and-simulation-methods-exam}


## Ozone concentration

We're going to look at the ozone data we worked on previously.

``` python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd

# Importation des données OZONE
data_ozone = pd.read_csv("ozone.txt",sep=";")
```

We wish to use the following regression model :
$$O_3=\beta_1+\beta_2 T_{12}+\beta_3V_x +\beta_4 Ne_{12} + \epsilon$$
with the use of a constant and 3 explicative variables :

-   the temperature at 12AM: $T_{12}$
-   the wind: $V_x$
-   the nebulosity level at 12AM: $Ne_{12}$



### Question 1: Display the summary of the results of a regression on this model with the summary() command from the package statsmodels 


```python
import statsmodels.api as sms

Y=data_ozone['O3']

X=sms.add_constant(data_ozone[['T12','Vx','Ne12']])
model=sms.OLS(Y,X)
results=model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     O3   R-squared:                       0.682
    Model:                            OLS   Adj. R-squared:                  0.661
    Method:                 Least Squares   F-statistic:                     32.87
    Date:                Mon, 23 May 2022   Prob (F-statistic):           1.66e-11
    Time:                        22:42:38   Log-Likelihood:                -200.50
    No. Observations:                  50   AIC:                             409.0
    Df Residuals:                      46   BIC:                             416.7
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         84.5473     13.607      6.214      0.000      57.158     111.936
    T12            1.3150      0.497      2.644      0.011       0.314       2.316
    Vx             0.4864      0.168      2.903      0.006       0.149       0.824
    Ne12          -4.8934      1.027     -4.765      0.000      -6.961      -2.826
    ==============================================================================
    Omnibus:                        0.211   Durbin-Watson:                   1.758
    Prob(Omnibus):                  0.900   Jarque-Bera (JB):                0.411
    Skew:                          -0.050   Prob(JB):                        0.814
    Kurtosis:                       2.567   Cond. No.                         148.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
 

### Question 2: Deduce from this summary the value of the 4 regression coefficients and the $R^2$.


<html>
<span style='color:green'>Your answer: B1 = 84.5473, B2 =
1.3150, B3 = 0.4864, B4 = -4.8934 :</span>
$$O_3= 84.5473 + 1.3150 * T_{12}+ 0.4864 * V_x - 4.8934 * Ne_{12}$$
<span style='color:green'> The R² is equal to 0.682, the model
explains 68% of the total variability of the data (which is not bad) </span>
</html>
 
### Question 3: Compute using the formula in class the estimation (non biased) of the std-deviation of the error, i.e. $\hat{\sigma}$. Find this result using the object that comes from your fitted model ( .fit())

```python
Ypredict = results.predict(X)
ResidualsError = Y-Ypredict
NbDonnees = len(data_ozone)
HatSigma2 = np.sum(ResidualsError**2)/(NbDonnees-4)

print("Non Biased estimator of the residual error's standard deviation :", HatSigma2**0.5)
print("Result found from the fit() method", results.mse_resid**0.5)
```

    Non Biased estimator of the residual error's standard deviation : 13.913257670973184
    Result found from the fit() method 13.913257670973184


<html>
<span style='color:green'>Your commentary : The std-deviation
is 13.91 for both methods. In the above table (Question 1) the value <tt>
F-Statistic</tt> is the value of the test statistic for the
global Fisher's test, i.e. we\'re testing $H_0:$ all of the
coefficients are equal to 0. Moreover <tt>
Prob(F-statistic)</tt> corresponds to the probability to
observe a statistic greater than the computed <tt>F-statistic</tt></span>
</html>

### Question 4: Compute using the formula the test statistic of the global Fisher Test and the value of the probability to find a test statistic greater than the computed F-statistic. What do you conclude regarding the null hypothesis ?

``` python
from scipy.stats import f

X2 = np.ones(len(data_ozone))
model2=sms.OLS(Y,X2)
results2=model2.fit()
Ypredict2=results2.predict(X2)

F_Statistic_Calcule= ((len(data_ozone) - 4)/ 3)*(np.sum((Ypredict-Ypredict2)**2))/np.sum((Y-Ypredict)**2)
Prob_F_Statistic_Calcule= f.cdf(F_Statistic_Calcule, (len(data_ozone) - 4), 3)


print(F_Statistic_Calcule)
print(1-Prob_F_Statistic_Calcule)
```


    32.86620920917683
    0.007239892209251697

<html>
<span style='color:green'>Your Commentary : We find the same
f-statistic in the summary() which is equal to 32.87 and the reject the
null hypothesis with a p-value smaller than 5%</span>
</html>

 

### Question 5: 

 - (a) What can you sat about the acceptance/reject of the hypothesis to have the $j$-th coefficient equal to 0 ? </span>

 - (b) Find the results <tt> t, P\>\|t\|, \[0.025 0.975\]</tt> for the coefficient linked to the variable $T_{12}$ with the formula seen in class. The values <tt> \[0.025 0.975\]</tt> correspond to the trust interval of $\beta_2$ at 95% trust.

<html> 
<span style='color:green'> Your answer : Given that we reject
the null hypothesis, all the coefficients are not equal to 0</span>
</html>


``` python
from scipy.stats import t #students distribution

Xmoy=np.mean(data_ozone.T12)
Sum_Xn=np.sum((data_ozone.T12-Xmoy)**2)

Ypredict = results.predict(X)
ResidualsError = Y-Ypredict
NbDonnees = len(data_ozone)
HatSigma2 = np.sum(ResidualsError**2)/(NbDonnees-4)

# std of the error of the second coefficient
Std2=np.sqrt(HatSigma2/Sum_Xn)
print("Standard Deviation of the error of the second coefficient's estimator:",Std2)

# Compute the t-stat: H0: Beta2=0
# Intercept:
print("T Test Statistic:",results.params[1]/Std2)

P_value = t.sf(results.params[1]/Std2, 4)
print("P-value of the t-statistic :",P_value, "We reject the null hypothesis : 'Béta2 = 0'")
```
    Standard Deviation of the error of the second coefficient's estimator: 0.425189722832548
    T Test Statistic: 3.0928449906563182
    P-value of the t-statistic : 0.018236714261109586 We reject the null hypothesis : 'Béta2 = 0'


 
### Question 6: We're going to do a sub-model test with the null hypothesis in the following nested model $$O_3=\beta_1+\beta_3V_x +\beta_4 Ne_{12} + \epsilon$$ against the complete model seen above. Compute the test statistic using the determination coefficients $R²$ and $R_a²$. Do we accept or reject the null hypothesis at the 5% level ? Is it equivalent to the former test ?

``` python
X3=sms.add_constant(data_ozone[['Vx','Ne12']])
model3=sms.OLS(Y,X3)
results3=model3.fit()

F = ((NbDonnees-4)/3)*(results.rsquared - results3.rsquared) / (1 - results.rsquared)

print(F)

P_value = t.sf(F, 4)
print("P-value of the test's statistic :",P_value, \
      "We reject the null hypothesis : 'The variable T12 does not add anything to the model'")
```


    2.329867603525416
    P-value of the test's statistic : 0.04013186575767187 We reject the null hypothesis : 'The variable T12 does not add anything to the model'

<html>
<span style='color:green'>Your commentary : We reject the null
hypothesis: \'the variable T12 brings nothing to the model\' at the 5%
level. We deduce that T12 is important to explain the ozone level. The
tests are equivalent because their test statistics are very close 2.64
for the constant alone and 2.33 without T12, the p-values are therefore
also very close</span>
</html>

 
## The height of the eucalyptus trees



### Question 7: We will now study the height data of eucalyptus trees. Propose a hypothesis test to decide between the 2 following nested models :

<html>
    <div style='color:red'> $$ ht=\beta_1+\beta_2 circ +\epsilon$$</div>
    <div style='color:red'> and </div>
    <div style='color:red'> $$ ht=\beta_1+\beta_2 circ + \beta_3 \sqrt{circ}+\epsilon$$</div>
    <div style='color:red'> These 2 models were studied in the previous lab. $ht$ and $circ$ correspond respectively to the height and circumference of the eucalyptus trees.</div>
</html>

``` python
data_euc = pd.read_csv("eucalyptus.txt",sep=";") # Import the data

import statsmodels.api as sms

YY = data_euc['ht']

XX1 = sms.add_constant(data_euc['circ'])
m1 = sms.OLS(YY,XX1)
fit1 = m1.fit()

data_euc['sqrtcirc'] = np.sqrt(data_euc.circ)
XX2 = sms.add_constant(data_euc[['circ','sqrtcirc']])
m2 = sms.OLS(YY,XX2)
fit2 = m2.fit()

print("R² of the second model :",fit2.rsquared,"R² of the first model :", fit1.rsquared)

# Fischer's test :

Nb = len(data_euc)

F = ((Nb-3)/2)*(fit2.rsquared - fit1.rsquared) / (1 - fit2.rsquared)

print("F-statistic :",F)

P_value = t.sf(F, 3)
print("P-value of the test statistic :",P_value, \
      "We reject the null hypothesis: 'the sqrtcirc variable does not add anything to the model' (by default the test has a significance level at 5%)")
```
    R² of the second model : 0.7921903882554493 R² of the first model : 0.7683202384330653
    F-statistic : 81.89908388010876
    P-value of the test statistic : 2.0061831358574297e-06 We reject the null hypothesis: 'the sqrtcirc variable does not add anything to the model' (by default the test has a significance level at 5%)

<html>
<span style='color:green'>Your comment: We conclude that the
model with sqrtcirc explains better the variability of the data. This is
explained by the increase of the R-squared in the second
model</span>
</html>
