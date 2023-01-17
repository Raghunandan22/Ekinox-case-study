#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installing required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[2]:


# Input datafile
df = pd.read_csv("student_data.csv")
df_o=df


# In[3]:


# Deleting unwanted columns & non actionable columns
df.drop(['FirstName','FamilyName','StudentID'], axis=1, inplace=True)
df.drop(['famrel','Medu','Fedu','Fjob', 'Mjob', 'guardian','reason'], axis=1, inplace=True)


# In[4]:


df.isnull().sum() # No nulls


# In[5]:


# unique value in each column
for col in df:
    print(df[col].unique())


# In[6]:


# Categorical encoding
binary = ["sex","address", "famsize", "Pstatus","nursery", "schoolsup", "famsup", "paid", "activities", "internet", "higher", "romantic"]


def binary_encoder(dataset, col):
    dataset[col] = dataset[col].astype('category')
    dataset[col] = dataset[col].cat.codes
    dataset[col] = dataset[col].astype('int')

for col in binary:
    binary_encoder(df, col)


# In[7]:


df.head()


# In[8]:


#Correlation plot
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,12))

ax = sns.heatmap(data=df.corr(), ax=ax, cmap="Blues")
ax.set_xlabel('Features',fontdict={"fontsize":16})
ax.set_ylabel('Features',fontdict={"fontsize":16})
ax.set_title('Correlation between different Features', loc="center", fontdict={"fontsize": 16, "fontweight":"bold"})

plt.savefig("heatmap.png", bbox_inches="tight")
plt.show()


# In[9]:


#Removing autocorrelated variables
df.drop(['Walc'], axis=1, inplace=True)


# In[10]:


# Separating data based on schools
df_gp = df[df['school']=='GP']
df_ms = df[df['school']=='MS']
# Separating data based on schools
df_gp2 = df[df['school']=='GP']
df_ms2 = df[df['school']=='MS']


# In[11]:


#Removing autocorrelated variables
df_gp.drop(['Student name'], axis=1, inplace=True)
df_ms.drop(['Student name'], axis=1, inplace=True)


# In[12]:


df_gp.drop(['school'], axis=1, inplace=True)
df_ms.drop(['school'], axis=1, inplace=True)


# In[13]:


# Separating dependent & indepdent variables for GP school
x_cols = df_gp.drop("FinalGrade", axis=1).columns
X_gp = df_gp[x_cols]
y_gp = df_gp["FinalGrade"]


# In[14]:


# Standardising and running regressor algorithm
sc = StandardScaler(with_mean=True, with_std=True)
X = pd.DataFrame(sc.fit_transform(X_gp), columns=x_cols)
X_train, X_test, y_train, y_test = train_test_split(X, y_gp, test_size=0.3, random_state=69)

rf = LinearRegression()#n_estimators=44, max_depth=10)
rf.fit(X_train, y_train)


pred_train_rf= rf.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))

pred_test_rf= rf.predict(X_test)

#rf.feature_importances_
rf.coef_
X_train.columns
# Creating dataset of features & their corresponding importance
dataset_gp = pd.DataFrame({'metric': X_train.columns, 'importance': rf.coef_}, columns=['metric', 'importance'])
dataset_gp


# In[15]:


df_gp2.head()


# In[16]:



from sklearn.preprocessing import MinMaxScaler

exclude_cols = ['school', 'FinalGrade', 'Student name']
scaler = MinMaxScaler()
cols_to_normalize = [col for col in df_gp2.columns if col not in exclude_cols]

# Normalize the columns
df_gp2[cols_to_normalize] = scaler.fit_transform(df_gp2[cols_to_normalize])
df_gp2.head()


# In[17]:


df_gp2.columns


# In[18]:


# Adding coefficients of each variable 
# Get the coefficient for the variable you want to use
# Keeping only actionable features
#coefficient_sex = dataset_gp.loc[dataset_gp['metric'] == 'sex', 'importance'].values[0]
#coefficient_age = dataset_gp.loc[dataset_gp['metric'] == 'age', 'importance'].values[0]
#coefficient_address = dataset_gp.loc[dataset_gp['metric'] == 'address', 'importance'].values[0]
#coefficient_famsize = dataset_gp.loc[dataset_gp['metric'] == 'famsize', 'importance'].values[0]
#coefficient_Pstatus = dataset_gp.loc[dataset_gp['metric'] == 'Pstatus', 'importance'].values[0]
#coefficient_traveltime = dataset_gp.loc[dataset_gp['metric'] == 'traveltime', 'importance'].values[0]
coefficient_studytime = dataset_gp.loc[dataset_gp['metric'] == 'studytime', 'importance'].values[0]
coefficient_failures = dataset_gp.loc[dataset_gp['metric'] == 'failures', 'importance'].values[0]
coefficient_schoolsup = dataset_gp.loc[dataset_gp['metric'] == 'schoolsup', 'importance'].values[0]
coefficient_famsup = dataset_gp.loc[dataset_gp['metric'] == 'famsup', 'importance'].values[0]
coefficient_paid = dataset_gp.loc[dataset_gp['metric'] == 'paid', 'importance'].values[0]
coefficient_activities = dataset_gp.loc[dataset_gp['metric'] == 'activities', 'importance'].values[0]
#coefficient_nursery = dataset_gp.loc[dataset_gp['metric'] == 'nursery', 'importance'].values[0]
coefficient_higher = dataset_gp.loc[dataset_gp['metric'] == 'higher', 'importance'].values[0]
coefficient_internet = dataset_gp.loc[dataset_gp['metric'] == 'internet', 'importance'].values[0]
coefficient_romantic = dataset_gp.loc[dataset_gp['metric'] == 'romantic', 'importance'].values[0]
coefficient_freetime = dataset_gp.loc[dataset_gp['metric'] == 'freetime', 'importance'].values[0]
coefficient_goout = dataset_gp.loc[dataset_gp['metric'] == 'goout', 'importance'].values[0]
coefficient_Dalc = dataset_gp.loc[dataset_gp['metric'] == 'Dalc', 'importance'].values[0]
coefficient_health = dataset_gp.loc[dataset_gp['metric'] == 'health', 'importance'].values[0]
coefficient_absences = dataset_gp.loc[dataset_gp['metric'] == 'absences', 'importance'].values[0]

# Create a new column in "df1" with the calculated values
#df_gp2['sex_importance'] = df_gp2['sex'] * coefficient_sex
#df_gp2['age_importance'] = df_gp2['age'] * coefficient_age
#df_gp2['address_importance'] = df_gp2['address'] * coefficient_address
#df_gp2['famsize_importance'] = df_gp2['famsize'] * coefficient_famsize
#df_gp2['Pstatus_importance'] = df_gp2['Pstatus'] * coefficient_Pstatus
#df_gp2['traveltime_importance'] = df_gp2['traveltime'] * coefficient_traveltime
df_gp2['studytime_importance'] = df_gp2['studytime'] * coefficient_studytime
df_gp2['failures_importance'] = df_gp2['failures'] * coefficient_failures
df_gp2['schoolsup_importance'] = df_gp2['schoolsup'] * coefficient_schoolsup
df_gp2['famsup_importance'] = df_gp2['famsup'] * coefficient_famsup
df_gp2['paid_importance'] = df_gp2['paid'] * coefficient_paid
df_gp2['activities_importance'] = df_gp2['activities'] * coefficient_activities
#df_gp2['nursery_importance'] = df_gp2['nursery'] * coefficient_nursery
df_gp2['higher_importance'] = df_gp2['higher'] * coefficient_higher
df_gp2['internet_importance'] = df_gp2['internet'] * coefficient_internet
df_gp2['freetime_importance'] = df_gp2['freetime'] * coefficient_freetime
df_gp2['romantic_importance'] = df_gp2['romantic'] * coefficient_romantic
df_gp2['goout_importance'] = df_gp2['goout'] * coefficient_goout
df_gp2['Dalc_importance'] = df_gp2['Dalc'] * coefficient_Dalc
df_gp2['health_importance'] = df_gp2['health'] * coefficient_health
df_gp2['absences_importance'] = df_gp2['absences'] * coefficient_absences


# In[19]:


df_gp2.tail()


# In[20]:


# Creating improvement score
df_gp2['Improvement_Score'] = df_gp2['studytime_importance']
#df_gp2['sex_importance'] 
#+df_gp2['age_importance'] 
#+df_gp2['address_importance'] 
#+df_gp2['famsize_importance'] 
#+df_gp2['Pstatus_importance'] 
#-df_gp2['traveltime_importance'] 
#+
+df_gp2['failures_importance'] 
+df_gp2['schoolsup_importance'] 
+df_gp2['famsup_importance'] 
+df_gp2['paid_importance'] 
+df_gp2['activities_importance'] 
#+df_gp2['nursery_importance'] 
+df_gp2['higher_importance'] 
+df_gp2['internet_importance'] 
+df_gp2['freetime_importance'] 
+df_gp2['romantic_importance'] 
+df_gp2['goout_importance'] 
+df_gp2['Dalc_importance'] 
+df_gp2['health_importance'] 
+df_gp2['absences_importance'] 


# In[21]:


gp_final =df_gp2[['school', 'FinalGrade','Student name','Improvement_Score']]
gp_final.head()


# In[22]:


#gp_final['Improvement_Score'] = (gp_final['Improvement_Score'] - gp_final['Improvement_Score'].min() )/ (gp_final['Improvement_Score'].max() - gp_final['Improvement_Score'].min())


# In[23]:


# Separating dependent & independent variables for MS school
x_cols = df_ms.drop("FinalGrade", axis=1).columns
X_ms = df_ms[x_cols]
y_ms = df_ms["FinalGrade"]


# In[24]:


# Standardising and running regressor algorithm
sc = StandardScaler(with_mean=True, with_std=True)
X = pd.DataFrame(sc.fit_transform(X_ms), columns=x_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y_ms, test_size=0.3, random_state=69)

rf = LinearRegression()#RandomForestRegressor(n_estimators=44, max_depth=10)
rf.fit(X_train, y_train)

pred_train_rf= rf.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))

pred_test_rf= rf.predict(X_test)

# Creating dataset of features & their corresponding importance
dataset_ms= pd.DataFrame({'metric': X_train.columns, 'importance': rf.coef_}, columns=['metric', 'importance'])
dataset_ms


# In[25]:



from sklearn.preprocessing import MinMaxScaler

exclude_cols = ['school', 'FinalGrade', 'Student name']
scaler = MinMaxScaler()
cols_to_normalize = [col for col in df_ms2.columns if col not in exclude_cols]

# Normalize the columns
df_ms2[cols_to_normalize] = scaler.fit_transform(df_ms2[cols_to_normalize])
df_ms2.head()


# In[26]:


# Adding coefficients of each variable 
# Get the coefficient for the variable you want to use
#coefficient_sex = dataset_ms.loc[dataset_ms['metric'] == 'sex', 'importance'].values[0]
#coefficient_age = dataset_ms.loc[dataset_ms['metric'] == 'age', 'importance'].values[0]
#coefficient_address = dataset_ms.loc[dataset_ms['metric'] == 'address', 'importance'].values[0]
#coefficient_famsize = dataset_ms.loc[dataset_ms['metric'] == 'famsize', 'importance'].values[0]
#coefficient_Pstatus = dataset_ms.loc[dataset_ms['metric'] == 'Pstatus', 'importance'].values[0]
#coefficient_traveltime = dataset_ms.loc[dataset_ms['metric'] == 'traveltime', 'importance'].values[0]
coefficient_studytime = dataset_ms.loc[dataset_ms['metric'] == 'studytime', 'importance'].values[0]
coefficient_failures = dataset_ms.loc[dataset_ms['metric'] == 'failures', 'importance'].values[0]
coefficient_schoolsup = dataset_ms.loc[dataset_ms['metric'] == 'schoolsup', 'importance'].values[0]
coefficient_famsup = dataset_ms.loc[dataset_ms['metric'] == 'famsup', 'importance'].values[0]
coefficient_paid = dataset_ms.loc[dataset_ms['metric'] == 'paid', 'importance'].values[0]
coefficient_activities = dataset_ms.loc[dataset_ms['metric'] == 'activities', 'importance'].values[0]
#coefficient_nursery = dataset_ms.loc[dataset_ms['metric'] == 'nursery', 'importance'].values[0]
coefficient_higher = dataset_ms.loc[dataset_ms['metric'] == 'higher', 'importance'].values[0]
coefficient_internet = dataset_ms.loc[dataset_ms['metric'] == 'internet', 'importance'].values[0]
coefficient_romantic = dataset_ms.loc[dataset_ms['metric'] == 'romantic', 'importance'].values[0]
coefficient_freetime = dataset_ms.loc[dataset_ms['metric'] == 'freetime', 'importance'].values[0]
coefficient_goout = dataset_ms.loc[dataset_ms['metric'] == 'goout', 'importance'].values[0]
coefficient_Dalc = dataset_ms.loc[dataset_ms['metric'] == 'Dalc', 'importance'].values[0]
coefficient_health = dataset_ms.loc[dataset_ms['metric'] == 'health', 'importance'].values[0]
coefficient_absences = dataset_ms.loc[dataset_ms['metric'] == 'absences', 'importance'].values[0]

# Create a new column in "df1" with the calculated values
#df_ms2['sex_importance'] = df_ms2['sex'] * coefficient_sex
#df_ms2['age_importance'] = df_ms2['age'] * coefficient_age
#df_ms2['address_importance'] = df_ms2['address'] * coefficient_address
#df_ms2['famsize_importance'] = df_ms2['famsize'] * coefficient_famsize
#df_ms2['Pstatus_importance'] = df_ms2['Pstatus'] * coefficient_Pstatus
#df_ms2['traveltime_importance'] = df_ms2['traveltime'] * coefficient_traveltime
df_ms2['studytime_importance'] = df_ms2['studytime'] * coefficient_studytime
df_ms2['failures_importance'] = df_ms2['failures'] * coefficient_failures
df_ms2['schoolsup_importance'] = df_ms2['schoolsup'] * coefficient_schoolsup
df_ms2['famsup_importance'] = df_ms2['famsup'] * coefficient_famsup
df_ms2['paid_importance'] = df_ms2['paid'] * coefficient_paid
df_ms2['activities_importance'] = df_ms2['activities'] * coefficient_activities
#df_ms2['nursery_importance'] = df_ms2['nursery'] * coefficient_nursery
df_ms2['higher_importance'] = df_ms2['higher'] * coefficient_higher
df_ms2['internet_importance'] = df_ms2['internet'] * coefficient_internet
df_ms2['freetime_importance'] = df_ms2['freetime'] * coefficient_freetime
df_ms2['romantic_importance'] = df_ms2['romantic'] * coefficient_romantic
df_ms2['goout_importance'] = df_ms2['goout'] * coefficient_goout
df_ms2['Dalc_importance'] = df_ms2['Dalc'] * coefficient_Dalc
df_ms2['health_importance'] = df_ms2['health'] * coefficient_health
df_ms2['absences_importance'] = df_ms2['absences'] * coefficient_absences


# In[27]:


# Creating improvement score
df_ms2['Improvement_Score'] =df_ms2['studytime_importance'] 
#df_ms2['sex_importance'] 
#+df_ms2['age_importance'] 
#+df_ms2['address_importance'] 
#+df_ms2['famsize_importance'] 
#+df_ms2['Pstatus_importance'] 
#-df_ms2['traveltime_importance'] 
+df_ms2['failures_importance'] 
+df_ms2['schoolsup_importance'] 
+df_ms2['famsup_importance'] 
+df_ms2['paid_importance'] 
+df_ms2['activities_importance'] 
#+df_ms2['nursery_importance'] 
+df_ms2['higher_importance'] 
+df_ms2['internet_importance'] 
+df_ms2['freetime_importance'] 
+df_ms2['romantic_importance'] 
+df_ms2['goout_importance'] 
+df_ms2['Dalc_importance'] 
+df_ms2['health_importance'] 
+df_ms2['absences_importance'] 


# In[28]:


ms_final =df_ms2[['school', 'FinalGrade','Student name','Improvement_Score']]
ms_final.head()


# In[29]:


# Creating dahsboard using dash package

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# define a function that returns the scatter plot figure
def create_scatter_plot(df):
    return px.scatter(df, x='FinalGrade', y='Improvement_Score', color='Student name')

app = dash.Dash()
server =app.server

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='scatter-plot-dropdown',
            options=[{'label': 'GP School', 'value': 'plot1'}, {'label': 'MS School', 'value': 'plot2'}],
            value='plot1',
            clearable=False
        )
    ]),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('scatter-plot-dropdown', 'value')]
)
def update_scatter_plot(selected_plot):
    if selected_plot == 'plot1':
        return create_scatter_plot(gp_final)
    else:
        return create_scatter_plot(ms_final)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
    
# renderer = app.renderer
# renderer.render(app.layout, "my_dashboard.html")


# In[30]:


# import plotly.express as px
# !pip install streamlit
# import streamlit as st


# In[31]:


# st.set_page_config(page_title ="Final grade v/s Improvement score",
#                   page_icon = ":bar_chart:",
#                   layout ="wide")


# In[32]:


# st.dataframe(gp_final)


# In[33]:


# st.sidebar.header("Please filter here -")
# city =st.sidebar.multiselect(
# "Select the city-",
# options = gp_final["Student name"].unique())


# In[34]:


# # import plotly.express as px
# # fig =px.scatter(gp_final,x='FinalGrade',y='Improvement_Score', color='Student name')
# # fig.show()
# # import plotly.graph_objects as go

# # fig = go.Figure(data=go.Scattergl(
# #     x = gp_final["FinalGrade"],
# #     y = gp_final["Improvement_Score"],
# #     mode='markers',
# #     marker=dict( 
# #         colorscale='Viridis',
# #         line_width=1
# #       )
# # ))
# # !pip install dash
# # !pip install dash_daq
# from dash import Dash, dcc, html, Input, Output
# import plotly.graph_objects as go
# import plotly.express as px
# import dash_daq as daq

# app = Dash(__name__)


# picker_style = {'float': 'left', 'margin': 'auto'}

# app.layout = html.Div([
#     html.H4('Interactive color picker with Dash'),
#     dcc.Graph(id="graph"),
#     daq.ColorPicker(
#         id='font', label='Font Color', size=150,
#         style=picker_style, value=dict(hex='#119DFF')),
#     daq.ColorPicker(
#         id='title', label='Title Color', size=150,
#         style=picker_style, value=dict(hex='#F71016')),
# ])

# @app.callback(
#     Output("graph", 'figure'), 
#     Input("font", 'value'),
#     Input("title", 'value'))
# def update_bar_chart(font_color, title_color):
#     df = gp_final # replace with your own data source
#     fig = go.Figure(px.scatter(
#         df, x="FinalGrade", y="Improvement_Score", height=350,
#         color="Student name", title="Final grade v/s Improvement score"))
#     fig.update_layout(
#         font_color=font_color['hex'],
#         title_font_color=title_color['hex'])
#     return fig

# app.run_server(debug=True)


# In[35]:


# import plotly.express as px
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# fig =px.scatter(gp_final,x='FinalGrade',y='Improvement_Score', color='Student name', title= "Final Grade v/s Improvement score - GS School")

# app1 = dash.Dash()
# app1.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])

# app1.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


# In[36]:


# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# fig2 =px.scatter(ms_final,x='FinalGrade',y='Improvement_Score', color='Student name', title= "Final Grade v/s Improvement score - MS School")

# app2 = dash.Dash()
# app2.layout = html.Div([
#     dcc.Graph(figure=fig2)
# ])

# app2.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


# In[37]:


# import plotly.express as px
# import dash
# import dash_core_components as dcc
# import dash_html_components as html

# import plotly.graph_objs as go
# #fig1 =px.scatter(gp_final,x='FinalGrade',y='Improvement_Score', color='Student name', title= "Final Grade v/s Improvement score - GS School")
# #fig2 =px.scatter(ms_final,x='FinalGrade',y='Improvement_Score', color='Student name', title= "Final Grade v/s Improvement score - MS School")
# app = dash.Dash()

# # Create a layout for the app
# app.layout = html.Div([
#     # First graph
#     dcc.Graph(
#         id='graph1',
#         figure={
#             'data': [
#                 go.Scatter(
#                     x=gp_final['FinalGrade'],
#                     y=gp_final['Improvement_Score'],
#                     mode='markers'
#                 )
#             ],
#             'layout': go.Layout(
#                 title='Final Grade v/s Improvement score - GS School',
#                 xaxis={'title': 'Final Grade'},
#                 yaxis={'title': 'Improvement score'}
#             )
#         }
#     ),
#     # Second graph
#     dcc.Graph(
#         id='graph2',
#         figure={
#             'data': [
#                 go.Scatter(
#                     x=ms_final['FinalGrade'],
#                     y=ms_final['Improvement_Score'],
#                     mode='markers'
#                 )
#             ],
#             'layout': go.Layout(
#                 title='Final Grade v/s Improvement score - MS School',
#                 xaxis={'title': 'Final Grade'},
#                 yaxis={'title': 'Improvement score'}
#             )
#         }
#     )
# ])

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True, use_reloader=False)


# In[ ]:




