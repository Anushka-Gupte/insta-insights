import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt

st.title('Instagram User Segmentation')
st.divider()
model =  pickle.load(open('insta-insights.pkl','rb'))

dataset = pd.read_csv('./data/Instagram visits clustering.csv')
X = dataset.iloc[:,[1,2]].values
y_kmeans = model.fit_predict(X)

fig,ax = plt.subplots()
ax.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 50, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 50, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 50, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 50, c = 'violet', label = 'Cluster 4')
ax.set_xlabel('Instagram Visit Score')
ax.set_ylabel('Spending Rank (0 to 100)')
ax.set_title('Clusters of Users')
ax.legend()
st.pyplot(fig)

st.markdown('###### Instagram Visit Score: A metric reflecting the frequency or intensity of Instagram visits.')
st.markdown('###### Spending Rank (0 to 100):A ranking of users spending behavior, normalized to a scale of 0 to 100.')

df = dataset.copy()
df['clusters'] = y_kmeans
cluster_stats = df.groupby('clusters').agg(
    user_count = ('clusters','count'),
    avg_visit_score = ('Instagram visit score','mean'),
    avg_spending_rank = ('Spending_rank(0 to 100)','mean')
).reset_index()

size = len(df)
cluster_stats['user %'] = ((cluster_stats['user_count'] / size) * 100).round(2)
cluster_stats1 = cluster_stats.rename(index={0:'Cluster 1',1: 'Cluster 2',2:'Cluster 3',3:'Cluster 4'})
st.subheader('User Clusters - Key Metrics')
cluster_stats1[['user_count','user %','avg_visit_score','avg_spending_rank']]


st.subheader('Interpretation')
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('### :green[Cluster 3 - Power Spenders]')
    st.markdown('**High visits** and **High Spending**')
    st.markdown('Target with **loyalty programs**')

with col2:
    st.markdown('### :red[Cluster 1 - Window Shoppers]')
    st.markdown('**High visits** and **Low Spending**')
    st.markdown('Target with **more discounts**')

with col3:
    st.markdown('### :blue[Cluster 2 - Occasional Buyers]')
    st.markdown('**Low visits** and **High Spending**')
    st.markdown('**Re-engage** these users')

with col4:
    st.markdown('### :violet[Cluster 4 - Low-Value Users]')
    st.markdown('**Low visits** and **Low Spending**')
    st.markdown('**Low Priority** for marketing investment')    
