# Fingertips_ml_projects_no1
Fingertips Machine learning project

<h1>Home Price Prediction & Analysis</h1>
<br>
<p>Here there 21 parameters are given from that we have to predict house price</p>
<br>
<p>Initially load csv data set into python</p>
<br>
<p> step 1 find the missing value ration for each columns   
  "Price","Bedroom2","Bathroom","Car","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude" columns have missing values
  
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

# for categorical columns Label encoding is done
# finaly combined both num_col and cat_col and make dataframe again for machine learning model creation
  
</p>

<p>
  I created one list of models lis=[[LinearRegression,Lasso,Ridge,DecisionTreeRegressor,RandomForestRegressor,GradientBoostingRegressor,KNeighborsRegressor,xg.XGBRegressor,ca.CatBoostRegressor]

  for model_class in lis:
    model=model_class()
    model.fit(x_train,y_train)
    
    model_name=model_class.__name__
    output_result["model"].append(model_name)
    print(f"Model Name:-->{model_name}")
    
    train_acc=round(model.score(x_train,y_train)*100,2)
    output_result["train_accuracy"].append(train_acc)
    print(f"Train_accuracy:-->{train_acc}")
    
    test_acc=model.score(x_test,y_test)
    output_result["test_accuracy"].append(test_acc)
    print(f"Test Accuracy:-->{test_acc}")
    
    y_pred= model.predict(x_test)
    
    MAE=mean_absolute_error(y_test,y_pred)
    output_result["Mean_a_error"].append(MAE)
    print(f"MAE:--->{MAE}")
    
    MSE=mean_squared_error(y_test,y_pred)
    output_result["Mean_square_error"].append(MSE)
    print(f"MSE:--->{MSE}")
    
    RMSE=np.sqrt(MSE)
    output_result["root_mean_square_error"].append(RMSE)
    print(f"RMSE:--->{RMSE}")
    
    R2= r2_score(y_test,y_pred)
    output_result["r2_error"].append(R2)
    print(f"R2 score:--->{R2}")
    print("\n\n")
    run above loops which return as all models 's train,test,accuracy,MAE,MSE,RMSE,R2_SCORE in dictionary form
    finally converted dictionary into dataframe for comaparing models and find best model
</p>

<p> After finding best model using RandomizedSearchCV method hyperparamter tuning done and best model with best parameter has been searched</p>

## detail coding is available in Fingertips_ml_project1.ipynb file
