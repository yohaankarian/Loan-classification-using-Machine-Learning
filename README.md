# LoanFit: Machine Learning based Loan Approval System
Creation of a prediction model to tackle and address a very common problem in the Banking industry. "To approve or to reject the loan application?" . Since there is a vast number of  applications being filed everyday, having a predictive model to help the executives to do their job by giving them a heads up about approval or rejection of a new loan application. 

The flow of the study is as follows:
1. Reading the loan data from a csv file after the data has been cleaned and wrangled using python.
2. Defining the problem statement and identifying the target variable.
3. Basic data exploration.
4. Rejecting useless column to avoid overfitting and reducing the number of features in the model.
5. Visual Exploratory data analysis for data discription using Histogram and Barcharts, identifying whether the target variable is categorical or qualitative.
6. Feature selection based on the data distribution.
7. Outlier treatment
8. Missing value treatment
9. Visual and statistical correlation analysis
10. Converting data to numeric or 1/0 for Machine Learning
11. Trying multiple classification algorithms to select the best model with the highest accuracy and least error
12. Finally, Deploying the best modal and developing an API for the same.
    
## Reading data into Python 
```Python 
# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')
#Reading the dataset
import pandas as pd
import numpy as np
from google.colab import files      #Importing the csv file into google colab and using pandas 
                                    # to store the data into a variable called Loandata
uploaded = files.upload()
import io
Loandata = pd.read_csv(io.BytesIO(uploaded['Loan_Approval_Data.csv']))
```
## Further cleaning of data, removing any duplicate rows
```Python
#Removing duplicate rows if any
print('Shape before deleting values: ', Loandata.shape)
Loandata = Loandata.drop_duplicates()
print('Shape after deleting duplicate values: ', Loandata.shape)

#Printing sample data
#Start observing the Quantitative / Categorical / Qualitative variables
Loandata.head(10)
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/e464eaf1-c457-4b0b-a392-eb1b6e89281f)

## Defining problem statement: 
Since the model to be created is one to predict whether to approve the loan application or not?
Target Variable = Loan_Status
Predictors: Gender, Married, Dependents, Education, Self_employed, Applicant income, Co applicant income, Loanamount, Loan_amount_term, Credit_history,Property_area

```Python
# Target Variable = Loan_status
#Predictors = Gender, Married, Dependents, Education, Self_employed, Applicant income, Co applicant income, Loanamount, Loan_amount_term, Credit_history,Property_area

#Loan_status = "N" means the loan was rejected
#Loan_status = "Y" means the loan was approved

#Target variable is categorical

%matplotlib inline
#Creating a bar chart as the target variable is categorical
GroupedData = Loandata.groupby('Loan_Status').size()
GroupedData.plot(kind = 'bar', figsize = (4,3))
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/6c281694-644e-4ab7-a5eb-06264ee3dfa0)

When performing classification, we make sure that there is a balance in the distribution of each class otherwise it affects the machine learning algorithm to learn all the classes

## Data exploration:
Here, we gauge the overall data. The volume of data, the types of columns present in the data. Initial assessment of the data should be done to identify which columns are Qualitative, Categorical or Qualitative.

This step helps in the column rejection process i.e, identify if a certain column will affect the target variable or not.

```Python
Loandata.info()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/52db45e1-eb34-4dd4-9f34-de8b65deed36)

Info() gives us a summarized information of the data, gives us a basic idea which columns have null values and which don't.

```Python
Loandata.describe(include='all')
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/2c012eb9-8816-424d-8c24-dbdec65d4d5b)

describe() gives us the descriptive statistics of the data

```Python
Loandata.nunique()
```
Using nunique() to find number of unique values in each column. We use this to understand whih columns are categorical and which is continuous.
Typically if the number of unique values are < 20 then the variable is likely to be categorical otherwise continuous. 

![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/38a26fc8-9e7c-4a09-bf53-4fa811e9187a)

Loan_Id , ApplicantIncome, CoapplicantIncome, LoanAmount-> high number of unique values -> Qualitative(Continuous)

Gender, Married , Dependents, Education, Self Employed, Credit_History, Property_ Area, Loan_ Status -> less number of unique values -> Categorical

## Results from data exploration:
We reject Loan_ID as this column does not affect the loan approval or rejection. The remaining columns need to be trated for missing values if there are any.

## Removing useless variables from the data:

```Python
UselessColumns = ['Loan_ID']
Loandata = Loandata.drop(UselessColumns,axis=1)
Loandata.head()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/2a94844c-909b-47bd-afe9-5bad635e6b97)

It can be seen that the unwanted column being Loan_ID has been removed.

## Visual Exploratory Data Analysis:
From the data exploration done, we have 8 categorical predictors in the data i.e;
1. Gender
2. Married
3. Dependents
4. Education
5. Self_Employed
6. Loan_Amount_Term
7. Credit_History
8. Property_Area

Now we use bar charts to see how the data is distributed for these categorical columns.

Creating a function to plot multiple bar charts at once for all the categorical variables.

```Python
def PlotBarCharts(inpData, colsToPlot):
    %matplotlib inline
    
    import matplotlib.pyplot as plt
    
    # Generating multiple subplots
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(40,6))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])

#####################################################################
# Calling the function
PlotBarCharts(inpData=Loandata, colsToPlot=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area'])
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/d9c508d4-f72c-4c70-9b87-844c2672863e)

## Bar Chart Interpretation
In this data, all the categorical columns except "Loan_Amount_Term" have satisfactory distribution for machine learning.

## Visualising distribution of all the continuous predictor variables in the data using histograms

Visualising the continuous predictor variables 'ApplicantIncome','CoapplicantIncome','LoanAmount'

```Python
Loandata.hist(['ApplicantIncome', 'CoapplicantIncome','LoanAmount'], figsize=(18,10))
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/ebfff15e-ab47-4b24-b9b1-523c3bd420f9)

## Histogram Interpretation:

Ideal outcome for the histogram is a bell curve or a slightly skewed bell curve. If there is too much skewness, then the outlier treatment should be done and the column should be re-examined, if that also does not solve the problem then reject the column.

**Selected Continuous Variables:**
1. ApplicantIncome: Outliers seen beyond 30000
2. CoapplicantIncome: Outliers seen boyond 15000
3. LoanAmount: Slightly skewed distribution

## Outlier treatment:
Outlier are extreme values in the data which are far away from most of the values. 

**Finding outliers for ApplicantIncome**
```Python
# Finding nearest values to 30000 mark for applicant income
Loandata['ApplicantIncome'][Loandata['ApplicantIncome']>20000].sort_values()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/744c3568-3261-4c19-ad76-fad759e0f178)

Replacing any value above 30000 with the nearest value i.e; 23803

```Python
Loandata['ApplicantIncome'][Loandata['ApplicantIncome']>30000] = 23803
```
**Finding outliers for CoapplicantIncome**
```Python
# Finding nearest values to 15000 mark
Loandata['CoapplicantIncome'][Loandata['CoapplicantIncome']>10000].sort_values()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/dac1fad3-bd9e-48fa-9b53-d5b3d0dfeaf4)

Replacing any value over 15000 with the nearest logical value 11300

```Python
Loandata['CoapplicantIncome'][Loandata['CoapplicantIncome']>15000] = 11300
```

## Visualizing distribution after outlier treatment:

```Python
Loandata.hist(['ApplicantIncome', 'CoapplicantIncome'], figsize=(18,5))
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/9b23d2c7-3ff0-40bc-9597-adcc6fc8dc2f)

## Missing value treatment:

If a column has more than 30% of the data missing, then the missing value treatment cannot be done. The column must be rejected because too much information is missing.

```Python
Loandata.isnull().sum()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/240aac3b-f379-4a2d-9a66-1f9aa03c8476)

Replacing the missing values with mode values for categorical columns and median values for Qualitative/ Continuous values

```Python
# Replacing the missing values
# Using MODE for categorical columns
Loandata['Gender'].fillna(Loandata['Gender'].mode()[0], inplace=True)
Loandata['Married'].fillna(Loandata['Married'].mode()[0], inplace=True)
Loandata['Dependents'].fillna(Loandata['Dependents'].mode()[0], inplace=True)
Loandata['Self_Employed'].fillna(Loandata['Self_Employed'].mode()[0], inplace=True)
# Using Mode value for Loan_Amount_Term since it is a categorical variable
Loandata['Loan_Amount_Term'].fillna(Loandata['Loan_Amount_Term'].mode()[0], inplace=True)
Loandata['Credit_History'].fillna(Loandata['Credit_History'].mode()[0], inplace=True)

# Using Median value for continuous columns
Loandata['LoanAmount'].fillna(Loandata['LoanAmount'].median(), inplace=True)
# Checking missing values again after the treatment
Loandata.isnull().sum()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/f825b37a-6553-4577-8fe0-79b092b95fc5)

Thus, we can see there are no more missing values.

## Feature Selection:
In this section, we choose the best columns which are correlated to the target variable.
This can be done by measuring the correlation values or ANOVA and Chi-Square tests.

## Box Plots:
Below is the box plot for the categorical target variable "Loan_Status" and the continuous predictors 
```Python
ContinuousColsList=['ApplicantIncome','CoapplicantIncome', 'LoanAmount']

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(ContinuousColsList), figsize=(18,5))

# Creating box plots for each continuous predictor against the Target Variable "Loan_Status"
for PredictorCol , i in zip(ContinuousColsList, range(len(ContinuousColsList))):
    Loandata.boxplot(column=PredictorCol, by='Loan_Status', figsize=(5,5), vert=True, ax=PlotCanvas[i])
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/15bbcdf6-41a8-44a1-8838-10e7634292e8)

As it can be seen, the distribution looks identical for all of the continuous variables i.e; the boxes start on the same line. Thus the variables are not correlated to each other. 

This can also be confirmed by using the ANOVA test, code of which is given below which has been defined in a function 

```Python
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print('##### ANOVA Results ##### \n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)
```

```Python
# Calling the function to check which categorical variables are correlated with target
ContinuousVariables=['ApplicantIncome', 'CoapplicantIncome','LoanAmount']
FunctionAnova(inpData=Loandata, TargetVariable='Loan_Status', ContinuousPredictorList=ContinuousVariables)
```

![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/f4317884-bac2-484e-830f-899eebe3b3ce)

## Grouped bar charts:
When the target variable is categorical and the predictor variables are also categorical, they can be visualised using bar charts and statistically using Chi-square test.

```Python
# Cross tablulation between two categorical variables
CrossTabResult=pd.crosstab(index=Loandata['Gender'], columns=Loandata['Loan_Status'])
CrossTabResult
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/89ecbea7-a4c7-43ed-97ce-5e6bc73d8999)

```Python
# Visual Inference using Grouped Bar charts
CategoricalColsList=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area']

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=len(CategoricalColsList), ncols=1, figsize=(10,50))

# Creating Grouped bar plots for each categorical predictor against the Target Variable "Loan_Status"
for CategoricalCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    CrossTabResult=pd.crosstab(index=Loandata[CategoricalCol], columns=Loandata['Loan_Status'])
    CrossTabResult.plot.bar(color=['red','blue'], ax=PlotCanvas[i])
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/cc09c845-5317-4674-ba9a-a46ce32e49e6)

If the ratio of bars is similar across all categories, then the two columns are not correlated
Thus, it can be seen that Married, Education, Credit_History and Property_Area is correlated with Loan_Status.

This can be seen with the Chi-Square test as well.

```Python
# Writing a function to find the correlation of all categorical variables with the Target variable
def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]

    for predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)

        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])

    return(SelectedPredictors)
```
```Python
CategoricalVariables=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area']

# Calling the function
FunctionChisq(inpData=Loandata,
              TargetVariable='Loan_Status',
              CategoricalVariablesList= CategoricalVariables)
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/8af203e7-0acd-4620-b9fb-06dd01d6fe96)

## Selecting final predictors for training the model
```Python
SelectedColumns=['Married', 'Education', 'Credit_History', 'Property_Area']

# Selecting final columns
DataForML=Loandata[SelectedColumns]
DataForML.head()
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/adcb107a-2b81-42ba-9323-1bf9e0a5ca75)

## Converting the binary nominal variable to numeric using 1/0 mapping:
```Python
# Treating the binary nominal variables first
DataForML['Married'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace=True)

# Looking at data after nominal treatment
DataForML.head()
```

## ML : Splitting the data into Training and Testing sample
70% of the data is randomly used for Training data and the rest 30% is used as Testing data.
```Python
# Separate Target Variable and Predictor Variables
TargetVariable='Loan_Status'
Predictors=['Married', 'Education', 'Credit_History', 'Property_Area_Rural',
       'Property_Area_Semiurban', 'Property_Area_Urban']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)
```
```Python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/f58706a7-da58-42c3-9751-d99101ed16cf)

## Logistic Regression
I have chosen to use logistic regression for training the model with penalty l2. Over-fitting tends to occur when the fitted model has many feature variables with relatively large weights in magnitude. To prevent this situation we use ridge regression. At a high level, in ridge regression, the loss function or the residual sum of squares is minimized by adding a shrinkage quantity. The ridge regression makes use of lambda, which acts as a tuning parameter for the model. As the value of the lambda increases, the coefficient estimates tend toward 0.

```Python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# choose parameter Penalty='l1' or C=1
# choose different values for solver 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
clf = LogisticRegression(C=1,penalty='l2', solver='newton-cg')

# Printing all the parameters of logistic regression
# print(clf)

# Creating the model on Training Data
LOG=clf.fit(X_train,y_train)
prediction=LOG.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(LOG, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))
```
Trained the model using the training data using logistic regression and constructed the confusion matrix, along with calculating the precision, accuracy, f1 score and support. Further, we perform 10- fold cross validation to determine the accuracy values.

We get the final average accuracy of the model to be 0.78

![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/68a4e482-fe7d-434d-927a-5ec8b292467f)

 ## Deployment of the model:
 Training the model with 100% of the data available and saving the file as a pickle file to be utilized everywhere.
 
```Python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# choose parameter Penalty='l1' or C=1
# choose different values for solver 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
clf = LogisticRegression(C=1,penalty='l2', solver='newton-cg')
# Training the model on 100% Data available
LogisticRegressionModel=clf.fit(X,y)
```
```Python
import pickle
import os

# Saving the Python objects as serialized files can be done using pickle library
# Here let us save the Final ZomatoRatingModel
with open('LogisticRegressionModel.pkl', 'wb') as fileWriteStream:
    pickle.dump(LogisticRegressionModel, fileWriteStream)
    # Don't forget to close the filestream!
    fileWriteStream.close()
    
print('pickle file of Predictive Model is saved at Location:',os.getcwd())
```

**Creating a python function to allow easy access**
```Python
def PredictLoanStatus(InputLoanDetails):
    import pandas as pd
    Num_Inputs=InputLoanDetails.shape[0]
    
    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input
    
    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForML.pkl')
    InputLoanDetails=InputLoanDetails.append(DataForML)
    
    # Treating the binary nominal variables first
    InputLoanDetails['Married'].replace({'Yes':1, 'No':0}, inplace=True)
    InputLoanDetails['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace=True)
    
    # Generating dummy variables for rest of the nominal variables
    InputLoanDetails=pd.get_dummies(InputLoanDetails)
            
    # Maintaining the same order of columns as it was during the model training
    Predictors=['Married', 'Education', 'Credit_History', 'Property_Area_Rural',
       'Property_Area_Semiurban', 'Property_Area_Urban']
    
    # Generating the input values to the model
    X=InputLoanDetails[Predictors].values[0:Num_Inputs]    
    

    
    # Loading the Function from pickle file
    import pickle
    with open('LogisticRegressionModel.pkl', 'rb') as fileReadStream:
        LogRegression_model=pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()
            
    # Genrating Predictions
    Prediction=LogRegression_model.predict(X)
    PredictedStatus=pd.DataFrame(Prediction, columns=['Predicted Status'])
    return(PredictedStatus)
  ```

## Testing of the model with unseen inputs:
```Python
# Calling the function for some loan applications
NewLoanApplications=pd.DataFrame(
data=[['No','Graduate',1,'Urban'],
     ['No','Graduate',0,'Urban'],['Yes','Not Graduate',1,'Rural'],['No','Graduate',0,'Rural']],
columns=['Married','Education','Credit_History','Property_Area'])

print(NewLoanApplications)

# Calling the Function for prediction
PredictLoanStatus(InputLoanDetails= NewLoanApplications)
```
![image](https://github.com/yohaankarian/Loan-classification-using-Machine-Learning/assets/76671049/40c3b995-dab2-4e8b-a621-7dd1236f7488)


## Conclusion: 
Thus, we can see that the model has predicted the loan status for the new unseen inputs and can determine if the loan can be given to the applicant or not based on the input parameters being:
1. Married
2. Education
3. Credit History
4. Property Area































