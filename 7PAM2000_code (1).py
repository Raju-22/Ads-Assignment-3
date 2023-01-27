#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pnd
import numpy as nmp
import matplotlib.pyplot as mtplt


# In[2]:


#Loading data
climate = pnd.read_csv("API_19_DS2_en_csv_v2_4756035.csv", skiprows =3)


# In[3]:


#Displaying it
climate.head()


# In[4]:


#Taking mean for various years and storing them in new columns to represent decade-wise data
column1 = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969']
column2 = ['1970','1971','1972','1973','1974','1975','1976','1977','1978','1979']
column3 = ['1980','1981','1982','1983','1984','1985','1986','1987','1988','1989']
column4 = ['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999']
column5 = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009']
column6 = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
column7 = ['2020','2021']
climate['1960s'] = climate[column1].mean(axis=1,skipna = True)
climate['1970s'] = climate[column2].mean(axis=1,skipna = True)
climate['1980s'] = climate[column3].mean(axis=1,skipna = True)
climate['1990s'] = climate[column4].mean(axis=1,skipna = True)
climate['2000s'] = climate[column5].mean(axis=1,skipna = True)
climate['2010s'] = climate[column6].mean(axis=1,skipna = True)
climate['2020s'] = climate[column7].mean(axis=1,skipna = True)
climate.head()


# In[5]:


#Dropping unnecessary columns
columns = ['1960','1961','1962','1963','1964','1965','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','Unnamed: 66','2021','Indicator Code','Country Code']
climate = climate.drop(columns=columns)
#Setting the column indicator name as index for data 
climate.set_index("Indicator Name",inplace = True)
#Filling null values with 0
climate.fillna(value = 0, axis=1, inplace = True)
climate.fillna(value = 0, axis=0, inplace = True)
climate.head(80)


# In[6]:


# Converting data type of the columns 
climate[['1960s','1970s','1980s','1990s','2000s','2010s','2020s']] = climate[['1960s','1970s','1980s','1990s','2000s','2010s','2020s']].astype(float)
climate


# In[7]:


#Taking out the data of 10 countries and sorting it
new_climate = climate[climate["Country Name"].isin(["United States","India","Japan","Australia","United Kingdom","China","France","Canada","Russia","Brazil","Germany"])]
new_climate.sort_values(by=['Country Name'], inplace = True)
new_climate.head()


# In[29]:


#Taking out the data related to Electricity production from renewable sources, excluding hydroelectric (% of total) for the chosen countries
climate_electricity = new_climate.loc['Electricity production from renewable sources, excluding hydroelectric (% of total)']
Clustering_climate_electricity=climate.loc['Electricity production from renewable sources, excluding hydroelectric (% of total)']
#resetting the index and making indicator name as a column again
climate_electricity.reset_index(level=0, inplace=True)
Clustering_climate_electricity.reset_index(level=0, inplace=True)
climate_electricity


# In[30]:


Clustering_climate_electricity


# In[9]:


#Taking out the data related to Electricity production from renewable sources, excluding hydroelectric (% of total) for the chosen countries
# Getting the data for the decade of 2010 and 1990
x = climate_electricity['2010s']
y = climate_electricity['1990s']
countries = ['Australia','United States','India','China','United Kingdom','Germany','France','Canada','Japan','Brazil']
mtplt.figure(figsize=(17,8))
#changing the scale of graph on both the axes
ax=mtplt.gca()
ax.locator_params('y', nbins=30)
mtplt.locator_params('x', nbins=20)
scatter = mtplt.scatter(x, y,s=50*climate_electricity['2010s'],c=climate_electricity['Country Name'].astype('category').cat.codes)
mtplt.legend(handles=scatter.legend_elements()[0], labels=countries, title="Countries", fontsize='12')
mtplt.xlabel("Electricity production from renewable sources in 2010s (%)", fontsize='15')
mtplt.ylabel("Electricity production from renewable sources in 1990s (%)", fontsize='15')
mtplt.title("Country-wise Electricity Production from Renewable Sources in %", fontsize='23')
mtplt.show()


# In[10]:


#making a transpose function
def Transpose_Dataset(dataframe,column):
    """This function takes in a dataframe and its column as arguments and makes a transpose of them."""
    dataframe = dataframe.loc["{}".format(column)]
    dataframe = dataframe.head(10)
    print("### Original Dataset ###")
    print(dataframe)
    #using numpy to generate transpose
    transpose = nmp.transpose(dataframe)
    transpose.columns = transpose.iloc[0]
    transpose = transpose.drop(transpose.index[0])
    return transpose


# In[11]:


#calling transpose function
transposed = Transpose_Dataset(new_climate,'Forest area (% of land area)')
print("\n\n### Transposed Dataset ###")
print(transposed)


# In[13]:


#plotting the transposed dataframe
transposed.plot(kind='barh', figsize=(10,8))
mtplt.legend(title="Country Name", bbox_to_anchor =(1, 1), fontsize='12')
mtplt.xlabel("Electric power consumption (kWh per capita)", fontsize='15')
mtplt.ylabel("Decades", fontsize='15')
mtplt.title("Electric power consumption (kWh per capita)", fontsize='20')


# # K-Means Clustering
# 

# In[32]:


Clustering_climate_electricity = Clustering_climate_electricity.drop('Indicator Name', axis =1)
Clustering_climate_electricity.head()


# In[33]:


Clustering_climate_electricity = Clustering_climate_electricity.drop('Country Name', axis=1)
Clustering_climate_electricity.head()


# In[72]:


# Create cluster feature
from sklearn.cluster import KMeans
import seaborn
'''Here created the function for the clustering which will divide the values in the different clusters and in this I have taken the 
3 clusters for all the values. First the values are fitting in the kmeans classifier and then divide the values according to their category.'''
def kmeansFunction():
    kmeans = KMeans(n_clusters=3)
    Clustering_climate_electricity["Cluster"] = kmeans.fit_predict(Clustering_climate_electricity)
    Clustering_climate_electricity["Cluster"] = Clustering_climate_electricity["Cluster"].astype("category")


# In[73]:


#plotting the graph for the cluster 
seaborn.relplot(
    x="1990s", y="2000s", hue="Cluster", data=Clustering_climate_electricity, height=5,);
mtplt.title("K-Means Clustering", fontsize=20)


# In[74]:


'''This cell is for the curve fit of the values and calculating the confidence range in the form of a graph. First, 
the function equator is defined for calculating the line of the curve fit, and then another function define as ask in the question
err ranges is defined for calculating the confidence range of the values. This cell is for the curve fit of the values.
Then, predicting the values based on the function, and finally, using the predicting values to produce the graph for the 
data points, confidence range, predicted value, and the line that provides the best match for the data.'''
#importing the modules for the curve fit
from scipy.optimize import curve_fit
from itertools import cycle, islice
import numpy as np
#Here providing the data for the curve fit
climate_electricityX = Clustering_climate_electricity['1990s']
climate_electricityY = Clustering_climate_electricity['2000s']

#Creating a equator functions for line generation
def equator(i,j,k,l):
    return l/(1 + np.exp(-j*(i-k)))

params, cov = curve_fit(equator, climate_electricityX, climate_electricityY)

#creating a function for calculating the confidence range
def err_ranges(i, y, params):
    elect_y_fit = equator(i, *params)
    elect_y_res = y - elect_y_fit
    sse = np.sum(elect_y_res**2)
    value = sse / (len(i) - len(params))
    
    ci = 1.96
    errRan = ci * np.sqrt(cov*value)
    return np.array([elect_y_fit, elect_y_res])

# creating the variables for prediction
elect_x_pred = 20
elect_y_pred = equator(elect_x_pred, *params)


#ploting a simple scattering points for the given data 
mtplt.plot(climate_electricityX, climate_electricityY, 'o', label='Data Points')
elect_x_fit = np.linspace(0,20,1000)
elect_y_fit = equator(elect_x_fit, *params)
#plotting for the line for the best fit
mtplt.plot(elect_x_fit, elect_y_fit, label='Best Fit')

rangeERR = err_ranges(elect_x_fit, elect_y_fit, params)
mtplt.fill_between(elect_x_fit, rangeERR[0], rangeERR[1], color='gray', alpha=0.5, label='Confidence Range')
mtplt.scatter(elect_x_pred, elect_y_pred, color='red', label='Predicted Value')

#giving the information of plot like labels and titles
mtplt.xlabel('1990',  fontsize=15)
mtplt.ylabel('2000', fontsize=15)
mtplt.title('Confidence ranges', fontsize=20)
mtplt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
mtplt.show()


# In[ ]:




