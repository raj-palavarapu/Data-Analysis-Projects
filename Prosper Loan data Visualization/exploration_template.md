
## Understanding the loan data


# Propser Data Exploration
## by Raj Palavarapu



> This notebook analyzes the loan data from Prosper.This data set contains 113,937 loans with 81 variables for each loan that has been issued.We will be analyzing the  various factors that are affecting the loan status and vizualize the relationship between various variables.

Variables that are used in the analysis
BorrowerAPR: The Borrower's Annual Percentage Rate (APR) for the loan.


CreditGrade: The Credit rating that was assigned at the time the listing went live. Applicable for listings pre-2009 period     and will only be populated for those listings.

OnTimeProsperPayments : Number of on time payments the borrower had made on Prosper loans at the time they created this listing. This value will be null if the borrower has no prior loans.

IncomeRange : The income range of the borrower at the time the listing was created.

StatedMonthlyIncome : The monthly income the borrower stated at the time the listing was created.

LoanOriginalAmount:The origination amount of the loan.

ProsperRating (Alpha):The Prosper Rating assigned at the time the listing was created between AA - HR.  Applicable for loans originated after July 2009.

DebtToIncomeRatio:The debt to income ratio of the borrower at the time the credit profile was pulled. This value is Null if the debt to income ratio is not available. This value is capped at 10.01 (any debt to income ratio larger than 1000% will be returned as 1001%).

LoanStatus:The current status of the loan: Cancelled,  Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue. The PastDue status will be accompanied by a delinquency bucket.

CreditGrade:The Credit rating that was assigned at the time the listing went live. Applicable for listings pre-2009 period and will only be populated for those listings.

EstimatedLoss:Estimated loss is the estimated principal loss on charge-offs. Applicable for loans originated after July 2009.

OnTimeProsperPayments:Number of on time payments the borrower had made on Prosper loans at the time they created this listing. This value will be null if the borrower has no prior loans.



```python
# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from IPython.display import display
%matplotlib inline
```

#### loading the data


```python
loan_data=pd.read_csv('data/prosperLoanData.csv')
display(loan_data.columns)
display(loan_data.info())
```


    Index(['ListingKey', 'ListingNumber', 'ListingCreationDate', 'CreditGrade',
           'Term', 'LoanStatus', 'ClosedDate', 'BorrowerAPR', 'BorrowerRate',
           'LenderYield', 'EstimatedEffectiveYield', 'EstimatedLoss',
           'EstimatedReturn', 'ProsperRating (numeric)', 'ProsperRating (Alpha)',
           'ProsperScore', 'ListingCategory (numeric)', 'BorrowerState',
           'Occupation', 'EmploymentStatus', 'EmploymentStatusDuration',
           'IsBorrowerHomeowner', 'CurrentlyInGroup', 'GroupKey',
           'DateCreditPulled', 'CreditScoreRangeLower', 'CreditScoreRangeUpper',
           'FirstRecordedCreditLine', 'CurrentCreditLines', 'OpenCreditLines',
           'TotalCreditLinespast7years', 'OpenRevolvingAccounts',
           'OpenRevolvingMonthlyPayment', 'InquiriesLast6Months', 'TotalInquiries',
           'CurrentDelinquencies', 'AmountDelinquent', 'DelinquenciesLast7Years',
           'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
           'RevolvingCreditBalance', 'BankcardUtilization',
           'AvailableBankcardCredit', 'TotalTrades',
           'TradesNeverDelinquent (percentage)', 'TradesOpenedLast6Months',
           'DebtToIncomeRatio', 'IncomeRange', 'IncomeVerifiable',
           'StatedMonthlyIncome', 'LoanKey', 'TotalProsperLoans',
           'TotalProsperPaymentsBilled', 'OnTimeProsperPayments',
           'ProsperPaymentsLessThanOneMonthLate',
           'ProsperPaymentsOneMonthPlusLate', 'ProsperPrincipalBorrowed',
           'ProsperPrincipalOutstanding', 'ScorexChangeAtTimeOfListing',
           'LoanCurrentDaysDelinquent', 'LoanFirstDefaultedCycleNumber',
           'LoanMonthsSinceOrigination', 'LoanNumber', 'LoanOriginalAmount',
           'LoanOriginationDate', 'LoanOriginationQuarter', 'MemberKey',
           'MonthlyLoanPayment', 'LP_CustomerPayments',
           'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees',
           'LP_CollectionFees', 'LP_GrossPrincipalLoss', 'LP_NetPrincipalLoss',
           'LP_NonPrincipalRecoverypayments', 'PercentFunded', 'Recommendations',
           'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount',
           'Investors'],
          dtype='object')


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 113937 entries, 0 to 113936
    Data columns (total 81 columns):
    ListingKey                             113937 non-null object
    ListingNumber                          113937 non-null int64
    ListingCreationDate                    113937 non-null object
    CreditGrade                            28953 non-null object
    Term                                   113937 non-null int64
    LoanStatus                             113937 non-null object
    ClosedDate                             55089 non-null object
    BorrowerAPR                            113912 non-null float64
    BorrowerRate                           113937 non-null float64
    LenderYield                            113937 non-null float64
    EstimatedEffectiveYield                84853 non-null float64
    EstimatedLoss                          84853 non-null float64
    EstimatedReturn                        84853 non-null float64
    ProsperRating (numeric)                84853 non-null float64
    ProsperRating (Alpha)                  84853 non-null object
    ProsperScore                           84853 non-null float64
    ListingCategory (numeric)              113937 non-null int64
    BorrowerState                          108422 non-null object
    Occupation                             110349 non-null object
    EmploymentStatus                       111682 non-null object
    EmploymentStatusDuration               106312 non-null float64
    IsBorrowerHomeowner                    113937 non-null bool
    CurrentlyInGroup                       113937 non-null bool
    GroupKey                               13341 non-null object
    DateCreditPulled                       113937 non-null object
    CreditScoreRangeLower                  113346 non-null float64
    CreditScoreRangeUpper                  113346 non-null float64
    FirstRecordedCreditLine                113240 non-null object
    CurrentCreditLines                     106333 non-null float64
    OpenCreditLines                        106333 non-null float64
    TotalCreditLinespast7years             113240 non-null float64
    OpenRevolvingAccounts                  113937 non-null int64
    OpenRevolvingMonthlyPayment            113937 non-null int64
    InquiriesLast6Months                   113240 non-null float64
    TotalInquiries                         112778 non-null float64
    CurrentDelinquencies                   113240 non-null float64
    AmountDelinquent                       106315 non-null float64
    DelinquenciesLast7Years                112947 non-null float64
    PublicRecordsLast10Years               113240 non-null float64
    PublicRecordsLast12Months              106333 non-null float64
    RevolvingCreditBalance                 106333 non-null float64
    BankcardUtilization                    106333 non-null float64
    AvailableBankcardCredit                106393 non-null float64
    TotalTrades                            106393 non-null float64
    TradesNeverDelinquent (percentage)     106393 non-null float64
    TradesOpenedLast6Months                106393 non-null float64
    DebtToIncomeRatio                      105383 non-null float64
    IncomeRange                            113937 non-null object
    IncomeVerifiable                       113937 non-null bool
    StatedMonthlyIncome                    113937 non-null float64
    LoanKey                                113937 non-null object
    TotalProsperLoans                      22085 non-null float64
    TotalProsperPaymentsBilled             22085 non-null float64
    OnTimeProsperPayments                  22085 non-null float64
    ProsperPaymentsLessThanOneMonthLate    22085 non-null float64
    ProsperPaymentsOneMonthPlusLate        22085 non-null float64
    ProsperPrincipalBorrowed               22085 non-null float64
    ProsperPrincipalOutstanding            22085 non-null float64
    ScorexChangeAtTimeOfListing            18928 non-null float64
    LoanCurrentDaysDelinquent              113937 non-null int64
    LoanFirstDefaultedCycleNumber          16952 non-null float64
    LoanMonthsSinceOrigination             113937 non-null int64
    LoanNumber                             113937 non-null int64
    LoanOriginalAmount                     113937 non-null int64
    LoanOriginationDate                    113937 non-null object
    LoanOriginationQuarter                 113937 non-null object
    MemberKey                              113937 non-null object
    MonthlyLoanPayment                     113937 non-null float64
    LP_CustomerPayments                    113937 non-null float64
    LP_CustomerPrincipalPayments           113937 non-null float64
    LP_InterestandFees                     113937 non-null float64
    LP_ServiceFees                         113937 non-null float64
    LP_CollectionFees                      113937 non-null float64
    LP_GrossPrincipalLoss                  113937 non-null float64
    LP_NetPrincipalLoss                    113937 non-null float64
    LP_NonPrincipalRecoverypayments        113937 non-null float64
    PercentFunded                          113937 non-null float64
    Recommendations                        113937 non-null int64
    InvestmentFromFriendsCount             113937 non-null int64
    InvestmentFromFriendsAmount            113937 non-null float64
    Investors                              113937 non-null int64
    dtypes: bool(3), float64(49), int64(12), object(17)
    memory usage: 68.1+ MB
    


    None



```python
loan_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ListingKey</th>
      <th>ListingNumber</th>
      <th>ListingCreationDate</th>
      <th>CreditGrade</th>
      <th>Term</th>
      <th>LoanStatus</th>
      <th>ClosedDate</th>
      <th>BorrowerAPR</th>
      <th>BorrowerRate</th>
      <th>LenderYield</th>
      <th>...</th>
      <th>LP_ServiceFees</th>
      <th>LP_CollectionFees</th>
      <th>LP_GrossPrincipalLoss</th>
      <th>LP_NetPrincipalLoss</th>
      <th>LP_NonPrincipalRecoverypayments</th>
      <th>PercentFunded</th>
      <th>Recommendations</th>
      <th>InvestmentFromFriendsCount</th>
      <th>InvestmentFromFriendsAmount</th>
      <th>Investors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1021339766868145413AB3B</td>
      <td>193129</td>
      <td>09:29.3</td>
      <td>C</td>
      <td>36</td>
      <td>Completed</td>
      <td>8/14/2009 0:00</td>
      <td>0.16516</td>
      <td>0.1580</td>
      <td>0.1380</td>
      <td>...</td>
      <td>-133.18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10273602499503308B223C1</td>
      <td>1209647</td>
      <td>28:07.9</td>
      <td>NaN</td>
      <td>36</td>
      <td>Current</td>
      <td>NaN</td>
      <td>0.12016</td>
      <td>0.0920</td>
      <td>0.0820</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0EE9337825851032864889A</td>
      <td>81716</td>
      <td>00:47.1</td>
      <td>HR</td>
      <td>36</td>
      <td>Completed</td>
      <td>12/17/2009 0:00</td>
      <td>0.28269</td>
      <td>0.2750</td>
      <td>0.2400</td>
      <td>...</td>
      <td>-24.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0EF5356002482715299901A</td>
      <td>658116</td>
      <td>02:35.0</td>
      <td>NaN</td>
      <td>36</td>
      <td>Current</td>
      <td>NaN</td>
      <td>0.12528</td>
      <td>0.0974</td>
      <td>0.0874</td>
      <td>...</td>
      <td>-108.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>158</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0F023589499656230C5E3E2</td>
      <td>909464</td>
      <td>38:39.1</td>
      <td>NaN</td>
      <td>36</td>
      <td>Current</td>
      <td>NaN</td>
      <td>0.24614</td>
      <td>0.2085</td>
      <td>0.1985</td>
      <td>...</td>
      <td>-60.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
loan_data.shape
```




    (113937, 81)



## Univariate Exploration

> In this section, investigate distributions of individual variables. If
you see unusual points or outliers, take a deeper look to clean things up
and prepare yourself to look at relationships between variables.


```python
loan_data.columns
```




    Index(['ListingKey', 'ListingNumber', 'ListingCreationDate', 'CreditGrade',
           'Term', 'LoanStatus', 'ClosedDate', 'BorrowerAPR', 'BorrowerRate',
           'LenderYield', 'EstimatedEffectiveYield', 'EstimatedLoss',
           'EstimatedReturn', 'ProsperRating (numeric)', 'ProsperRating (Alpha)',
           'ProsperScore', 'ListingCategory (numeric)', 'BorrowerState',
           'Occupation', 'EmploymentStatus', 'EmploymentStatusDuration',
           'IsBorrowerHomeowner', 'CurrentlyInGroup', 'GroupKey',
           'DateCreditPulled', 'CreditScoreRangeLower', 'CreditScoreRangeUpper',
           'FirstRecordedCreditLine', 'CurrentCreditLines', 'OpenCreditLines',
           'TotalCreditLinespast7years', 'OpenRevolvingAccounts',
           'OpenRevolvingMonthlyPayment', 'InquiriesLast6Months', 'TotalInquiries',
           'CurrentDelinquencies', 'AmountDelinquent', 'DelinquenciesLast7Years',
           'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
           'RevolvingCreditBalance', 'BankcardUtilization',
           'AvailableBankcardCredit', 'TotalTrades',
           'TradesNeverDelinquent (percentage)', 'TradesOpenedLast6Months',
           'DebtToIncomeRatio', 'IncomeRange', 'IncomeVerifiable',
           'StatedMonthlyIncome', 'LoanKey', 'TotalProsperLoans',
           'TotalProsperPaymentsBilled', 'OnTimeProsperPayments',
           'ProsperPaymentsLessThanOneMonthLate',
           'ProsperPaymentsOneMonthPlusLate', 'ProsperPrincipalBorrowed',
           'ProsperPrincipalOutstanding', 'ScorexChangeAtTimeOfListing',
           'LoanCurrentDaysDelinquent', 'LoanFirstDefaultedCycleNumber',
           'LoanMonthsSinceOrigination', 'LoanNumber', 'LoanOriginalAmount',
           'LoanOriginationDate', 'LoanOriginationQuarter', 'MemberKey',
           'MonthlyLoanPayment', 'LP_CustomerPayments',
           'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees',
           'LP_CollectionFees', 'LP_GrossPrincipalLoss', 'LP_NetPrincipalLoss',
           'LP_NonPrincipalRecoverypayments', 'PercentFunded', 'Recommendations',
           'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount',
           'Investors'],
          dtype='object')



> Checking the distribution of Borrower APR, the distribution is almost normal


```python
loan_data['BorrowerAPR'].describe()
```




    count    113912.000000
    mean          0.218828
    std           0.080364
    min           0.006530
    25%           0.156290
    50%           0.209760
    75%           0.283810
    max           0.512290
    Name: BorrowerAPR, dtype: float64




```python
bins = np.arange(0, loan_data['BorrowerAPR'].max()+0.05, 0.02)
plt.figure(figsize=[10, 5])
plt.hist(data = loan_data, x = 'BorrowerAPR', bins = bins);
plt.xlabel('Borrower APR');
```


![png](output_12_0.png)



```python
loan_data['LoanOriginalAmount'].describe()
```




    count    113937.00000
    mean       8337.01385
    std        6245.80058
    min        1000.00000
    25%        4000.00000
    50%        6500.00000
    75%       12000.00000
    max       35000.00000
    Name: LoanOriginalAmount, dtype: float64



> Distrubution of loan amount, it looks like the distribution is right skewed

>Distrubution of propser ratings


```python
loan_data['ProsperRating (Alpha)'].value_counts()
```




    C     18345
    B     15581
    A     14551
    D     14274
    E      9795
    HR     6935
    AA     5372
    Name: ProsperRating (Alpha), dtype: int64




```python
order=loan_data['ProsperRating (Alpha)'].value_counts().index
base_color = sb.color_palette()[0]
sb.countplot(data = loan_data, x = 'ProsperRating (Alpha)', color = base_color,order=order)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c3be7b8>




![png](output_17_1.png)


> Distribution of credit Grade


```python
loan_data['CreditGrade'].value_counts()
```




    C     5649
    D     5153
    B     4389
    AA    3509
    HR    3508
    A     3315
    E     3289
    NC     141
    Name: CreditGrade, dtype: int64




```python
order=loan_data['CreditGrade'].value_counts().index
plt.figure(figsize=[10,10])
base_color = sb.color_palette()[0]
sb.countplot(data = loan_data, x = 'CreditGrade', color = base_color,order=order)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f5706f85c0>




![png](output_20_1.png)


> Distribution of debt to income ratio


```python
loan_data['DebtToIncomeRatio'].describe()
```




    count    105383.000000
    mean          0.275947
    std           0.551759
    min           0.000000
    25%           0.140000
    50%           0.220000
    75%           0.320000
    max          10.010000
    Name: DebtToIncomeRatio, dtype: float64




```python
bins = np.arange(0,loan_data['DebtToIncomeRatio'].max()+0.2,0.1)
plt.figure(figsize=[10, 5])
plt.hist(data = loan_data, x = 'DebtToIncomeRatio', bins = bins);
plt.xlabel('Debt to income Ratio');
```


![png](output_23_0.png)


> Distribution of Borrower's income


```python
bins = np.arange(0,50000,500)
plt.figure(figsize=[10, 5])
plt.hist(data = loan_data, x = 'StatedMonthlyIncome', bins = bins);
plt.xlabel('Borrowers Monthly Income');
```


![png](output_25_0.png)



```python
#The distribution of borrowers monthly income looks right skewed, percent of borrowers whose  monthly income greater than 35k
```


```python
pct=(loan_data['StatedMonthlyIncome']>35000).sum()/float(loan_data.shape[0])
pct
```




    0.0018167934911398405



Less than 0.2 percent borrowers have stated monthly income greater than 35k and this looks like an outlier

> Distribution of Loan status


```python
loan_data['LoanStatus'].value_counts()
```




    Current                   56576
    Completed                 38074
    Chargedoff                11992
    Defaulted                  5018
    Past Due (1-15 days)        806
    Past Due (31-60 days)       363
    Past Due (61-90 days)       313
    Past Due (91-120 days)      304
    Past Due (16-30 days)       265
    FinalPaymentInProgress      205
    Past Due (>120 days)         16
    Cancelled                     5
    Name: LoanStatus, dtype: int64




```python
order=loan_data['LoanStatus'].value_counts().index
plt.figure(figsize=[25,15])
base_color = sb.color_palette()[0]
sb.countplot(data = loan_data, x = 'LoanStatus', color = base_color,order=order)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c865048>




![png](output_31_1.png)



```python
bins = np.arange(0,loan_data['EstimatedLoss'].max()+0.1,0.1)
plt.figure(figsize=[8,5])
plt.hist(data = loan_data, x = 'EstimatedLoss', bins = bins);
plt.xlabel('Estimated Loss');
```


![png](output_32_0.png)


## Bivariate Exploration



> We will analyze the relationship between CreditGrade amd Borrower APR and see how creditGrade is affecting the APR



```python
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]
sb.boxplot(data = loan_data, x = 'CreditGrade', y = 'BorrowerAPR', color = base_color)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c3869b0>




![png](output_35_1.png)


> We will analyze the relationship between CreditGrade amd LoanOriginalAmount and see how creditGrade is affecting the Loanamount


```python
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]
sb.boxplot(data = loan_data, x = 'CreditGrade', y = 'LoanOriginalAmount', color = base_color)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c8d18d0>




![png](output_37_1.png)


>We will analyze the relationship between ProsperRating amd BorrowerAPR and see how ProsperRating is affecting the BorrowerAPR


```python
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]
sb.boxplot(data = loan_data, x = 'ProsperRating (Alpha)', y = 'BorrowerAPR', color = base_color)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c8d7128>




![png](output_39_1.png)


>We will analyze the relationship between ProsperRating amd OnTimeProsperPayments and see if highprosper rating is resulting in more onTime payments


```python
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]
sb.boxplot(data = loan_data, x = 'ProsperRating (Alpha)', y = 'OnTimeProsperPayments', color = base_color)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f56c8d1198>




![png](output_41_1.png)


> Top 5 stated with highest Defualted loans


```python
base_color = sb.color_palette()[0]
ax=sb.countplot(y='BorrowerState',data=loan_data,color=base_color,order=loan_data['BorrowerState'].value_counts().iloc[:5].index)
```


![png](output_43_0.png)


## Multivariate Exploration

> In this Multivariate Exploration we are going to analyze the Propser Rating effect on relationship of APR and loan amount


```python
g=sb.FacetGrid(data=loan_data, aspect=1.5, height=5, col='ProsperRating (Alpha)',col_wrap=4)
g.map(sb.regplot, 'LoanOriginalAmount', 'BorrowerAPR', x_jitter=0.03, scatter_kws={'alpha':0.1});
g.add_legend();
```


![png](output_45_0.png)


It looks like the loan amount increases with better prosper rating.Looking at the above Grid closely it looks like people with better prosper rating tend to borrow more because they might be getting a good APR and people with less propser rating tend to borrow less due to high APR
