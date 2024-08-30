# app to analyse the top 100 baby names over time in NSW

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel
import streamlit as st
import matplotlib.pyplot as plt


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

###### app text ######

st.header("Welcome to the name checker app!")
st.subheader("Want to avoid your child or pet having the same name as everyone else's? You're in the right place.")
st.text("This app allows you to check how popular a name is and whether it is likely to become more popular. Scroll down to input a name.")

# read in the data
file = 'popular_baby_names_1952_to_2023.csv'
df = pd.read_csv(file)

# make all the names lower case
df['Name'] = df['Name'].str.lower()

# group the data
grouped = df.groupby('Name')

# what name do you want to check? 
# ask the user to input a name
name = st.text_input("What name do you want to use?", "George")
st.write("The current name is", name)

#gender = st.text_input("What gender statistics do you want to see?", "Male")
#st.write("The current gender is", gender)

# make the name lower case
name = name.lower()

# check if the name is in the list
if name in grouped.groups:
    st.write('That name is in our list - checking now!')
else:
    st.write('That name isnt in the top 100, safe to use!')

# get the data for the name
name_data = grouped.get_group(name)

# Alternatively, if you need to filter by both name and gender:
#name_data = grouped[(grouped['Name'] == name) & (grouped['Gender'] == gender)]

# run the gaussian process regression
x_future, x_full, y_full_mean, y_full_sigma  = run_gaussian_process_regression(name_data)

# recapitalise the name for output 
display_name = name.capitalize()

tab1, tab2, tab3 = st.tabs(["Statistics", "Graph", "Predictions"])
with tab1:
    st.header(display_name+" statistics")
    # get the stats for the selected name 
    # check whether the name made the list in 2023
    if 2023 in name_data['Year'].values:
        # what was the ranking of the name in 2023
        rank_2023 = name_data.loc[name_data['Year'] == 2023]['Rank'].values[0]
        # how many of that name were there in 2023
        number_2023 = name_data.loc[name_data['Year'] == 2023]['Number'].values[0]
        
        st.write(display_name + " was ranked " + rank_2023 + " in 2023 with " + number_2023 + "babies given this name.")
    else:
        # when was the last year the name was in the top 100
        year_last = name_data['Year'].max()
        # what was the ranking of the name in the last year
        rank_last = name_data.loc[name_data['Year'] == last_year]['Rank'].values[0]

        st.write(display_name+" was last in the top 100 in " + year_last + " when it was ranked " + rank_last + ".")

    # average number of that name per year
    mean = name_data['Number'].mean()

    st.write("The average number of babies named " + display_name + " per year is " + mean + ".")

    # it was most popular in which year
    max_year = name_data.loc[name_data['Number'].idxmax()]['Year']

    st.write(display_name + " was most popular in " + max_year + ".")

    # what was the highest ranking of that name
    max_rank = name_data.loc[name_data['Number'].idxmax()]['Rank']

    st.write("When it was ranked " + max_rank + ".")

with tab2:
    st.header(display_name+" over time")
    # plot the name prevalence over time 
    plt.figure()

    fig, ax = plt.subplots()

    ax.plot(name_data['Year'], name_data['Number'], color='black')

    ax.set_xlabel('Year')
    ax.set_ylabel('Number')
    ax.legend()

    st.pyplot(fig)


with tab3:
    st.header("Predictions for "+display_name)
    # plot the name prevalence over time with the fit and prediction
    plt.figure()

    fig, ax = plt.subplots()

    ax.plot(name_data['Year'], name_data['Number'], label='Data', color='black')
    ax.plot(x_full, y_full_mean, 'b-', label='Prediction')
    ax.fill_between(x_full.ravel(), y_full_mean - 1.96 * y_full_sigma, 
                y_full_mean + 1.96 * y_full_sigma, alpha=0.2, color='blue')

    ax.set_xlabel('Year')
    ax.set_ylabel('Number')
    ax.legend()

    st.pyplot(fig)



