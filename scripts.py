import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import SAtom2 as sa



def transform_categorical_features(XY_raw, verbose=True):
  """
  Transforms categorical features to numerical using ordinal encoder.
      
  Parameters
  ----------
      XY_raw : DataFrame, input/output matrix

  Returnts
  ----------
      XY : DataFrame, transformed i/o matrix with numerical features
    
  """  

  # Obtain labels for categorical features (if present)
  ord_attr = XY_raw.select_dtypes(exclude=['float64', 'int64']).columns

  if len(ord_attr) == 0:
    print(f'No categorical features detected.')
    XY = XY_raw.copy()

  else:
    print(f'Found categorical features: {ord_attr}')

    # Pipeline for ordinal encoder
    ordinal_pipeline = Pipeline([
                                # ('imputer', SimpleImputer(strategy="most_frequent")),
                                ('encode', OrdinalEncoder()),
                                ])

    # Full pipeline, transform categorical features and update dataframe
    full_pipeline = ColumnTransformer([("ord", ordinal_pipeline, ord_attr),],
                                      remainder='passthrough')
    XY_arr = full_pipeline.fit_transform(XY_raw)

    XY = pd.DataFrame(XY_arr, columns=ord_attr.to_list() + XY_raw.columns.drop(ord_attr).to_list())
    if verbose:
      display(XY.head())
    
  return XY

def create_new_samples(X_pdfs, N):
  """
  Sample N new inputs from probability densitity distributions (PDF).
  Distributions are limited to uniform, continuous and uniform, discrete.
      
  Parameters
  ----------
      X_pdfs : dictionary, decription of PDF for each variable input in X
      N : int, number of new samples
  """  
  # Initialize array for new samples and list for inputs labels
  x_arr = np.zeros([N, len(X_pdfs)])
  x_labels = []

  # Loop through dictionary - each describing variable input, Xi
  for i, (xvar, nest_dict) in enumerate(X_pdfs.items()):
    x_labels = x_labels + [nest_dict['param']]

    if nest_dict['pdf'] == 'uniform':
      x_temp = np.random.uniform(nest_dict['args'][0], nest_dict['args'][1], N)
    elif nest_dict['pdf'] == 'discrete':
      x_temp = np.random.choice(nest_dict['args'], N)

    x_arr[:,i] = np.reshape(x_temp,-1)
  
  X_new = pd.DataFrame(x_arr, columns=x_labels)
  return X_new


def print_R2_performance(model, test=False):
    """
    Print R2 score for training and validation set
    
    Requires: x_train_prepared, x_valid_prepared, y_train, y_valid
    
    Parameters
    ----------
        model : scikit-learn model
        test : boolean, True to include performance on test set
    """
    y_train_pred = model.predict(x_train_full.values)
    # y_valid_pred = model.predict(x_valid_prepared)
    r2_train = r2_score(y_train_full, y_train_pred)
    # r2_valid = r2_score(y_valid, y_valid_pred)
#     message = f'R²(train) = {r2_train:.3f} \t R²(valid) = {r2_valid:.3f}'
    message = f'# {r2_train:.3f}'#' / {r2_valid:.3f}'
    
    if test:
        y_test_pred = model.predict(x_test.values)
        r2_test = r2_score(y_test, y_test_pred)
        message = message + f' / {r2_test:.3f}'
    print(message)

def make_predictions(model):
  y_train_pred = model.predict(x_train_prepared)
  y_train_full_pred = model.predict(x_train_full_prepared)
  y_valid_pred = model.predict(x_valid_prepared)
  y_test_pred = model.predict(x_test_prepared)
  return y_train_pred, y_train_full_pred, y_valid_pred, y_test_pred


def sa_multiple(X, Y, J=50, include_SA_all=False, sort_by=None, figsize='auto'):
  """Perform sensitivity analysis for each output in matrix using SAtom

  Parameters:
  -----------
  X (pandas.DataFrame): DataFrame with numeric input values
  Y (pandas.DataFrame): DataFrame with numeric output values
  J : int, default=50
    Number of repeated randomly selected samples. 
  include_SA_all : {True, False}, default=False
    If True, TOM SA will also be performed for all outputs
  sort_by : {None, 'alphabetically', 'all'} or output label
    Decide how to sort the input labels on bar plots, e.g. using SA values.  
  figsize : {'auto', (float, float)} figure width and height in inches. 'auto'
    will set the figure size automatically from the number of inputs and outputs  

  Returns:
  --------
  Y_SA (pandas.DataFrame): indexed with input names and columns with SA metrics
    for each output
  fig (matplotlib.Figure): plot figure

  References
  ----------
  Østergård, T., Jensen, R.L., and Maagaard, S.E. (2017)
      Interactive Building Design Space Exploration Using Regionalized 
      Sensitivity Analysis, 15th conference of the International Building 
      Performance Simulation Association, 7-9 August 2017, San Francisco, USA  

  """
  
  n_outputs = Y.shape[1] + 1 if include_SA_all else Y.shape[1]
  n_inputs = Y.shape[1]
  Y_SA = np.zeros([X.shape[1], Y.shape[1]])

  # Loop and perform SA for each output
  for i, col in enumerate(Y.columns):
    print(f'({i+1}/{n_outputs}) {"Performs TOM SA for:" :>20}  {col}')
    tom = sa.TOM(X, Y[col], J=J, verbose=False, dummy=False)
    KS_means = np.reshape(np.transpose(tom.KS_df.tail(1).values),-1)
    KS_means = KS_means / sum(KS_means) * 100 # Convert to percentages
    Y_SA[:, i] = KS_means

  # Convert array to DataFrame
  Y_SA = pd.DataFrame(data=Y_SA, index=list(X.columns.values), columns=Y.columns)
  
  if include_SA_all:
    print(f'({n_outputs}/{n_outputs}) {"Performs TOM SA for:" :>20}  All outputs')
    tom = sa.TOM(X, Y, J=J*2, verbose=False, dummy=False)
    KS_means = np.reshape(np.transpose(tom.KS_df.tail(1).values),-1)
    KS_means = KS_means / sum(KS_means) * 100 # Convert to percentages    
    Y_SA.insert (0, "All", KS_means)

  # Sort inputs by output(s)
  if sort_by != None:
    if sort_by == 'alphabetically':
      Y_SA.sort_index(axis=0, inplace=True)
    if sort_by == 'all':
      if include_SA_all == True:
        Y_SA = Y_SA.sort_values(by='All', ascending=False) 
    if sort_by in Y_SA.columns:
      Y_SA = Y_SA.sort_values(by=sort_by, ascending=False) 

  # Plot distributions and horisontal barplots with SA results
  print('Preparing plots')
  
  if figsize == 'auto': figsize = (2+n_outputs*1.7, 2+n_inputs*.3)
  fig, axes = plt.subplots(2, n_outputs, 
                           figsize=figsize,  # (2+n_outputs*1.7, 1+n_inputs*1)
                           constrained_layout=True, gridspec_kw={'height_ratios': [1, 3]},
                           ) #  sharey='row',

  for i_plot in range(n_outputs):
    # Create histograms (except for "all outputs")
    if i_plot == 0 and include_SA_all == True:
      axes[0,0].axis('off')
    else:
      sns.histplot(ax=axes[0, i_plot], data=Y, x=Y_SA.columns[i_plot], 
                 stat="probability", element="step", fill=True, bins=25)
    
    # Create SA bar plots            
    sns.barplot(ax=axes[1, i_plot], y=Y_SA.index, x=Y_SA.iloc[:,i_plot])
    axes[1, i_plot].spines['top'].set_visible(False)  
    axes[1, i_plot].spines['right'].set_visible(False)  

    # Add labels
    for i, val in enumerate(Y_SA.iloc[:,i_plot].values):
        axes[1, i_plot].text(val+0.1, i+0.2, str(round(val,1)), size=8)

    axes[0, i_plot].set_title(Y_SA.columns[i_plot], fontsize=10)
    axes[0, i_plot].xaxis.label.set_visible(False)
    if i_plot >= 1:
      axes[0, i_plot].get_yaxis().set_visible(False)
      axes[1, i_plot].get_yaxis().set_visible(False)
    else:
      axes[0, i_plot].yaxis.set_ticks([])

  return Y_SA, fig


import math
round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))

def obtain_pdfs(X, n_significant_figures=3, verbose=True):
  """
  Obtain pdfs for each column in an input sample matrix, X.
  Assumes discrete distribution if number of unique values are less than 20, 
  otherwise a continuous, uniform distribution 
  
  Parameters
  ----------
    X {DataFrame} : Monte Carlo input values
    n_significat_figures : integer to round min/max values for uniform PDF

  Returns:
  --------
    X_pdfs {dict} : nested dict for each column in X
      e.g. 'x1' : {'idx':0, 'param': 'myvar', 'pdf': 'uniform', 'args': [0 2]}
  """
  X_pdfs = {}
  n_max_letters = len(max(X.columns, key=len)) + 2 # For optimal space in print
  for i, col in enumerate(X.columns):
    if len(X[col].unique()) <= 20:
      pdf = 'discrete'
      args = X[col].unique()
    else:
      pdf = 'uniform'      
      args = [round_to_n(X[col].min(), n_significant_figures), 
              round_to_n(X[col].max(), n_significant_figures)]
    
    X_pdfs['x'+str(i)] = {'idx':i, 'param':col, 'pdf':pdf, 'args':args}    
    
    if verbose:
      print(f'x{str(i):<2} {col:<{n_max_letters}} {pdf:<10} {args}')

  return X_pdfs


def ordinal_decode_cat_hyperparameters(cvres_df):
  """
  Ordinal decode categorical columns in grid search results' dataframe.
  Afterwards, sensitivity analysis can be performed to see which hyperparameters
  seems to affect the RMSE performance the most.
  
  Parameters
  ----------
    cvres_df {DataFrame} : Grid-search values and corresponding RMSE score

  Returns:
  --------
    cvres_df
  """
  numerical_attr = cvres_df.select_dtypes(include=['float64', 'int64']).columns
  ord_attr = cvres_df.select_dtypes(exclude=['float64', 'int64']).columns

  ordinal_pipeline = Pipeline([('encode', OrdinalEncoder())])
  small_pipeline = ColumnTransformer([("ord", ordinal_pipeline, ord_attr)], 
                                    remainder='passthrough')
  cvres_df = small_pipeline.fit_transform(cvres_df)

  # Convert transformed array to frame and reinsert RMSE as first column
  cvres_df = pd.DataFrame(cvres_df, columns=list(ord_attr) + list(numerical_attr))
  rmse_column = cvres_df.pop('RMSE')
  cvres_df.insert(0, 'RMSE', rmse_column)

  return cvres_df