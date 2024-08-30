# app to analyse the top 100 baby names over time in NSW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel
import streamlit as st


###### functions #######

def run_gaussian_process_regression(name_data, projection_years=20):
    # Extract Year and Count
    X = name_data['Year'].values[::-1].reshape(-1, 1)
    y = name_data['Number'].values[::-1]

    # Define the kernel: Constant * RBF
    # kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))

    # Define the kernel: Constant * RBF + ExpSineSquared
    # kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1)) + ExpSineSquared(1.0, 5.0, (1e-4, 1e1), (1e-4, 1e1))

    # Define the kernel: Constant + RBF + Periodic + WhiteKernel (for noise)
    kernel = C(1.0, (1e-4, 1e1)) + RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1)) + ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))


    # Create GaussianProcessRegressor object
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit to the data
    gp.fit(X, y)

    # Make predictions
    y_pred, sigma = gp.predict(X, return_std=True)

    # Extend the Year range for projection
    last_year = name_data['Year'].max()
    x_future = np.arange(last_year + 1, last_year + projection_years + 1).reshape(-1, 1)
    x_full = np.vstack((X, x_future))

    # Make predictions for the future years
    y_full_mean, y_full_sigma = gp.predict(x_full, return_std=True)

    return x_future, x_full, y_full_mean, y_full_sigma



###### end functions ####### 

# read in the data
file = 'popular_baby_names_1952_to_2023.csv'
df = pd.read_csv(file)

# make all the names lower case
df['Name'] = df['Name'].str.lower()

# group the data
grouped = df.groupby('Name')

# what name do you want to check? 
# ask the user to input a name
name = input('What do you want to call your baby? ')

name = st.text_input(
    "What name do you want to use?",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
    )

# make the name lower case
name = name.lower()

# check if the name is in the list
if name in grouped.groups:
    print('That name is in our list - checking now!')
else:
    print('That name isnt in the top 100, safe to use!')

# get the data for the name
name_data = grouped.get_group(name)

# run the gaussian process regression
x_future, x_full, y_full_mean, y_full_sigma  = run_gaussian_process_regression(name_data)

# plot the name prevalence over time and the fit
plt.figure()
sns.lineplot(x=name_data['Year'], y=name_data['Number'], label='Data')
plt.plot(x_full, y_full_mean, 'b-', label='Prediction')
plt.fill_between(x_full.ravel(), y_full_mean - 1.96 * y_full_sigma, 
                  y_full_mean + 1.96 * y_full_sigma, alpha=0.2, color='blue')
plt.xlabel('Year')
plt.ylabel('Number')
plt.legend()
plt.show()