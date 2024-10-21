# app to analyse the top 100 baby names over time in NSW

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel, Matern
import streamlit as st
import matplotlib.pyplot as plt

 
# TO DO:
# - implement a gender checker of some kind to catch names that appear in both gender lists 
# - add page for top 10 by year 
# - account for trailing spaces 
# - if name isnt found ask whether they mean a similar name
# - GP kernel sufficient?
# - check what the average is doing - over all years or the ones it is in the dataset for?


###### functions #######

def run_gaussian_process_regression(name_data, projection_years=20):
    # Extract Year and Count
    X = name_data['Year'].values[::-1].reshape(-1, 1)
    y = name_data['Number'].values[::-1]

    

    # Define kernel parameters
    length_scale = 10.0
    variance = y.std()
    avg = np.median(y)

    # Define the kernel: Constant + RBF + Periodic + WhiteKernel (for noise)
    kernel = avg + C(200, (100, 500)) * RBF(length_scale=50.0, length_scale_bounds=(20, 500))  + WhiteKernel(noise_level=1, noise_level_bounds=(1, 10))


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

st.set_page_config(page_title="Name Check 📈")
st.title("Name Check 📈")
st.header(":red[Welcome to the Name Check App!]")
st.subheader("Need to name a child? Want to know more about your name? You're in the right place!")
#st.markdown("As a Rebecca born in the 90s I am acutely aware of *how annoying* it can be to have an extremely common name. With this app you can check how **common prospective names are at the moment**, how common they were in the **past**, and whether they are about to be **trending**.")

st.text("This app is maintained by Rebecca McElroy and uses census data from the NSW Government.")
st.page_link("https://rebeccamcelroy.github.io/", label="Rebecca's Homepage", icon="🏠")

# read in the aus data
file = 'popular_baby_names_1952_to_2023.csv'
df = pd.read_csv(file)

# read in the us data

# loop over all the files in directory names_us
n = 0
for i in range(1880, 2024):
    file = f'./data/names_us/yob{i}.txt'
    df_temp = pd.read_csv(file, header=None)

    # add the year to the dataframe
    df_temp['Year'] = i

    # add the index modulo 1000 to the dataframe
    df_temp['Rank'] = (df_temp.index + 1) % 1000

    # add the data to the main dataframe
    if n == 0:
        df_us = df_temp

    else:
        df_us = pd.concat([df_us, df_temp])

    n += 1

# change the names of the columns
# Name, Gender, Number, Year
df_us.rename(columns = {df_us.columns[0]: 'Name'}, inplace=True)
df_us.rename(columns = {df_us.columns[1]: 'Gender'}, inplace=True)
df_us.rename(columns = {df_us.columns[2]: 'Number'}, inplace=True)

tab_aus, tab_us = st.tabs(["Australia - NSW", "USA"])

with tab_aus:

    tab_check, tab_history = st.tabs(["Check a name", "Check a year"])

    with tab_check:

        # make all the names lower case
        df['Name'] = df['Name'].str.lower()

        # group the data
        grouped = df.groupby('Name')

        # what name do you want to check? 
        # ask the user to input a name
        name = st.text_input("What name do you want to check?", "James")
        st.write(f"Currently checking :red[{name}]")

        #gender = st.text_input("What gender statistics do you want to see?", "Male")
        #st.write("The current gender is", gender)

        # make the name lower case
        name = name.lower()

        # check if the name is in the list
        if name in grouped.groups:
            st.write('That name is in the top 100. Scroll down for statistics, graphs, and predictions...')

            # get the data for the name
            name_data = grouped.get_group(name)

            # Alternatively, if you need to filter by both name and gender:
            #name_data = grouped[(grouped['Name'] == name) & (grouped['Gender'] == gender)]

            # run the gaussian process regression
            x_future, x_full, y_full_mean, y_full_sigma  = run_gaussian_process_regression(name_data)

            # recapitalise the name for output 
            display_name = name.capitalize()

            # use dark theme
            plt.style.use('dark_background')

            tab1, tab2, tab3 = st.tabs(["Statistics", "Graph", "Predictions"])
            with tab1:
                st.header("Statistics")
                # get the stats for the selected name 
                # check whether the name made the list in 2023
                if 2023 in name_data['Year'].values:
                    # what was the ranking of the name in 2023
                    rank_2023 = name_data.loc[name_data['Year'] == 2023]['Rank'].values[0]
                    # how many of that name were there in 2023
                    number_2023 = name_data.loc[name_data['Year'] == 2023]['Number'].values[0]
                    
                    st.write(f":red[{display_name}] was ranked :red[{rank_2023}] in 2023, :red[{number_2023}] babies were called this." )
                
                else:
                    # when was the last year the name was in the top 100
                    year_last = name_data['Year'].max()
                    # what was the ranking of the name in the last year
                    rank_last = name_data.loc[name_data['Year'] == year_last]['Rank'].values[0]
                
                    st.write(f":red[{display_name}] was last in the top 100 in {year_last} when it was ranked {rank_last}.")
                
                # total number of babies with that name
                total = name_data['Number'].sum()

                # how many unique years are there in the dataset?
                unique_years = name_data['Year'].nunique()

                # average number of babies with that name per year
                n_years = 72 # 1952-2023
                mean = total / 72 

                st.write(f"On average :red[{int(mean)}] babies were named :red[{display_name}] in NSW each year over the past {n_years} years.")

                # it was most popular in which year
                max_year = name_data.loc[name_data['Rank'].idxmin()]['Year']
                # what was the highest ranking of that name
                max_rank = name_data.loc[name_data['Rank'].idxmin()]['Rank']
                
                st.write(f":red[{display_name}] was most popular in {max_year}, when it was ranked {max_rank}.")
                
                # future stats
                # what does the model think 
                    
                if 2023 in name_data['Year'].values:
                    st.write(f"⚠️ :red[Warning, prediction optimizer in flux:]")
                        # future stats
                    # how many babies are predicted to be named that in 2033
                    future_number = y_full_mean[-10]
                
                    # difference between 2023 and 2033
                    ratio = future_number/number_2023
                
                    # what does this ratio mean?
                    # if ratio is approximately 1 then the name is likely to stay the same
                    # if ratio is less than 1 then the name is likely to decrease in popularity
                    # if ratio is greater than 1 then the name is likely to increase in popularity
                
                    if ratio > 0.9 and ratio < 1.1:
                        st.write(f"The number of babies named {display_name} in the future is likely to stay about the same.")
                
                    elif ratio < 0.9:
                        st.write(f"The number of babies named :red[{display_name}] in the future is likely to decrease.")
                
                    elif ratio > 1.1:
                        st.write(f"The number of babies named :red[{display_name}] in the future is likely to increase.")
                
                    #st.write(f":red[R = {ratio}]")
                
                else:
                    st.write(f":red[{display_name}] wasn't in the top 100 names last year, should be safe to use. ")
                
                
                        


            with tab2:
                st.header(display_name+" over time")
                # plot the name prevalence over time 
                plt.figure()
                
                fig, ax = plt.subplots()
                
                ax.plot(name_data['Year'], name_data['Number'], color='white')
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number')
                
                st.pyplot(fig)


            with tab3:
                
                st.header("Predictions for "+display_name)
                # plot the name prevalence over time with the fit and prediction
                plt.figure()
                
                fig, ax = plt.subplots()
                
                ax.plot(name_data['Year'], name_data['Number'], label='Data', color='white')
                ax.plot(x_full, y_full_mean, 'lightcoral', label='Prediction')
                #ax.fill_between(x_full.ravel(), y_full_mean - 1.96 * y_full_sigma, 
                #            y_full_mean + 1.96 * y_full_sigma, alpha=0.2, color='blue')
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number')
                ax.legend()
                
                st.pyplot(fig)


        else:
            st.write('That name has never been in the top 100, it must be unique 😲')

    with tab_history: 

        file = 'popular_baby_names_1952_to_2023.csv'

        year_select = st.text_input("What year do you want to check?", "2023")
        st.write(f"Currently checking :red[{year_select}]")

        # split the data into male and female
        df_male = df[df['Gender'] == 'Male']
        df_female = df[df['Gender'] == 'Female']


        # grab the top 10 male names for the selected year
        df_male_top = df_male[df_male['Year'] == int(year_select)]
        df_female_top = df_female[df_female['Year'] == int(year_select)]

        df_male_top = df_male_top.sort_values('Rank', ascending=True)
        df_female_top = df_female_top.sort_values('Rank', ascending=True)

        df_male_top = df_male_top.head(10)
        df_female_top = df_female_top.head(10)


        tab_female, tab_male = st.tabs(["Female names", "Male names"])

        with tab_female:

            # display the top ten names 
            st.write(f"Top 10 female names for the year {year_select}")
            st.dataframe(df_female_top[['Rank', 'Name', 'Number']], hide_index=True) 

        with tab_male:
            st.write(f"Top 10 male names for the year {year_select}")
            st.dataframe(df_male_top[['Rank', 'Name', 'Number']], hide_index=True) 

with tab_us:

    tab_check, tab_history, tab_pick = st.tabs(["Check a name", "Check a year", "Pick a name"])

    with tab_check:

        # names are in first column
        # make all the names lower case
        df_us['Name'] = df_us['Name'].str.lower()

        # group the data 
        grouped = df_us.groupby('Name')

        # what name do you want to check? 
        # ask the user to input a name
        name = st.text_input("What name do you want to check?", "Harvey")
        st.write(f"Currently checking :red[{name}]")

        #gender = st.text_input("What gender statistics do you want to see?", "Male")
        #st.write("The current gender is", gender)

        # make the name lower case
        name = name.lower()

        # check if the name is in the list
        if name in grouped.groups:
            st.write('That name is in the top 1000. Scroll down for statistics, graphs, and predictions...')

            # get the data for the name
            name_data = grouped.get_group(name)

            # Alternatively, if you need to filter by both name and gender:
            #name_data = grouped[(grouped['Name'] == name) & (grouped['Gender'] == gender)]

            # run the gaussian process regression
            x_future, x_full, y_full_mean, y_full_sigma  = run_gaussian_process_regression(name_data)

            # recapitalise the name for output 
            display_name = name.capitalize()

            # use dark theme
            plt.style.use('dark_background')

            tab1, tab2, tab3 = st.tabs(["Statistics", "Graph", "Predictions"])
            with tab1:
                st.header("Statistics")
                # get the stats for the selected name 
                # check whether the name made the list in 2023
                if 2023 in name_data['Year'].values:
                    # calculate the ranking of the name in 2023
                    # isolate name data for 2023
                    name_data_2023 = name_data[name_data['Year'] == 2023]
                    # calculate the rank of the name in 2023
                    rank_2023 = name_data_2023['Rank'].values[0]
                    # calculate the number of babies with that name in 2023
                    number_2023 = name_data.loc[name_data['Year'] == 2023]['Number'].values[0]
                    
                    st.write(f":red[{display_name}] was ranked :red[{rank_2023}] in 2023, :red[{number_2023}] babies were called this." )
                
                else:
                    # when was the last year the name was in the top 100
                    year_last = name_data['Year'].max()
                    # what was the ranking of the name in the last year
                    rank_last = name_data.loc[name_data['Year'] == year_last]['Rank'].values[0]
                
                    st.write(f":red[{display_name}] was last in the top 1000 in {year_last} when it was ranked {rank_last}.")
                
                # total number of babies with that name
                total = name_data['Number'].sum()

                # how many unique years are there in the dataset?
                unique_years = name_data['Year'].nunique()

                # average number of babies with that name per year
                n_years = 2023-1880 # 1952-2023
                mean = total / n_years

                st.write(f"On average :red[{int(mean)}] babies were named :red[{display_name}] in the US each year over the past {n_years} years.")

                # it was most popular in which year
                max_year = name_data.loc[name_data['Number'].idxmin()]['Year']
                # what was the highest ranking of that name
                # max_ = name_data.loc[name_data['Rank'].idxmin()]['Rank']
                
                st.write(f":red[{display_name}] was most popular in {max_year}.")
                
                # future stats
                # what does the model think 
                    
                if 2023 in name_data['Year'].values:
                    st.write(f"⚠️ :red[Warning, prediction optimizer in flux:]")
                        # future stats
                    # how many babies are predicted to be named that in 2033
                    future_number = y_full_mean[-10]
                
                    # difference between 2023 and 2033
                    ratio = future_number/number_2023
                
                    # what does this ratio mean?
                    # if ratio is approximately 1 then the name is likely to stay the same
                    # if ratio is less than 1 then the name is likely to decrease in popularity
                    # if ratio is greater than 1 then the name is likely to increase in popularity
                
                    if ratio > 0.9 and ratio < 1.1:
                        st.write(f"The number of babies named {display_name} in the future is likely to stay about the same.")
                
                    elif ratio < 0.9:
                        st.write(f"The number of babies named :red[{display_name}] in the future is likely to decrease.")
                
                    elif ratio > 1.1:
                        st.write(f"The number of babies named :red[{display_name}] in the future is likely to increase.")
                
                    #st.write(f":red[R = {ratio}]")
                
                else:
                    st.write(f":red[{display_name}] wasn't in the top 100 names last year, should be safe to use. ")
                
                
                        


            with tab2:
                st.header(display_name+" over time")
                # plot the name prevalence over time 
                plt.figure()
                
                fig, ax = plt.subplots()
                
                ax.plot(name_data['Year'], name_data['Number'], color='white')
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number')
                
                st.pyplot(fig)


            with tab3:
                
                st.header("Predictions for "+display_name)
                # plot the name prevalence over time with the fit and prediction
                plt.figure()
                
                fig, ax = plt.subplots()
                
                ax.plot(name_data['Year'], name_data['Number'], label='Data', color='white')
                ax.plot(x_full, y_full_mean, 'lightcoral', label='Prediction')
                #ax.fill_between(x_full.ravel(), y_full_mean - 1.96 * y_full_sigma, 
                #            y_full_mean + 1.96 * y_full_sigma, alpha=0.2, color='blue')
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number')
                ax.legend()
                
                st.pyplot(fig)


        else:
            st.write('That name has never been in the top 100, it must be unique 😲')

    with tab_history: 

        file = 'popular_baby_names_1952_to_2023.csv'

        year_select = st.text_input("What year do you want to check?", "2023")
        st.write(f"Currently checking :red[{year_select}]")

        # split the data into male and female
        df_male = df[df['Gender'] == 'Male']
        df_female = df[df['Gender'] == 'Female']


        # grab the top 10 male names for the selected year
        df_male_top = df_male[df_male['Year'] == int(year_select)]
        df_female_top = df_female[df_female['Year'] == int(year_select)]

        df_male_top = df_male_top.sort_values('Rank', ascending=True)
        df_female_top = df_female_top.sort_values('Rank', ascending=True)

        df_male_top = df_male_top.head(10)
        df_female_top = df_female_top.head(10)


        tab_female, tab_male = st.tabs(["Female names", "Male names"])

        with tab_female:

            # display the top ten names 
            st.write(f"Top 10 female names for the year {year_select}")
            st.dataframe(df_female_top[['Rank', 'Name', 'Number']], hide_index=True) 

        with tab_male:
            st.write(f"Top 10 male names for the year {year_select}")
            st.dataframe(df_male_top[['Rank', 'Name', 'Number']], hide_index=True) 

    with tab_pick:

        st.write("WIP - we will pick a random uncommon name for you to consider")


# with tab4:
#     st.header("Predictions for "+display_name)
#     # plot the name prevalence over time with the fit and prediction
#     #fig = px.line(name_data, x='Year', y="Number")

#     fig = go.Figure()

#     fig = px.line(name_data, x='Year', y='Number')

#     fig.add_scatter(x=x_full, y=y_full_mean, name='Model', mode='lines')

#     st.plotly_chart(fig, theme="streamlit")





