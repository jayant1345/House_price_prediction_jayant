# Fingertips_ml_projects_no1
Fingertips Machine learning project

<h1>Home Price Prediction & Analysis</h1>
<br>
<p>Here there 21 parameters are given from that we have to predict house price</p>
<br>
<p>Initially load csv data set into python</p>
<br>
<p> step 1 find the missing value ration for each columns   "Price","Bedroom2","Bathroom","Car","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude" columns have missing values
  
 <b>num_col=data.select_dtypes(exclude="object")
  cat_col=data.select_dtypes(include="object")</b>

so num_col and cat_col are seperated

  <b>num_col1=pd.DataFrame(num_col)
  num_col=pd.DataFrame(imputer.fit_transform(num_col1),columns=num_col.columns)</b>
<br>
above listed colummns have more than 10% missing values so using sklearn KNN impurter we impute missing values
<br>
### capping method using IQR
<b>def outlier_removal(data,columns):
    
    for col in columns:
        feature=data[col]
        q1=feature.quantile(0.25)
        q3=feature.quantile(0.75)
        IQR=q3-q1
        lower_lim=float(q1-1.5*IQR)
        upper_lim=float(q3+1.5*IQR)
        
        ## REMOVING OUTLIERS
        data[col]=np.where(data[col]< lower_lim,lower_lim,data[col])
        data[col]=np.where(data[col]> upper_lim,upper_lim,data[col])
        
    return data</b>
using above function removed outliers and capped outliers

## for categorical columns Label encoding is done
# finaly combined both num_col and cat_col and make dataframe again for machine learning model creation
  
</p>
