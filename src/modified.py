import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import string
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import re
from yellowbrick.cluster import KElbowVisualizer

#Read Scrapped Text
df = pd.read_csv('./files/text_translated.csv')

#####################Text Processing
def text_process(text):
    '''
    Performs the following given input text:
    - Removal of non-ASCII characters (non-translated foreign characters)
    - Removal of punctuation
    - Tokenization & removal of whitespace
    - Conversion to lowercase
    - Removal of stop words
    - Lemmatization
    '''
    stemmer = WordNetLemmatizer()
    #Remove non-ASCII characters (non-translated foreign characters)
    text = text.encode("ascii", errors="ignore").decode()
    #Remove numbers
    #text = re.sub(r'\d+', '', text)
    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Tokenize & remove whitespace
    tokens = word_tokenize(text)
    #Remove stop words & convert to lowercase
    nostop =  [word.lower() for word in tokens if word not in stopwords.words('english')]
    #Lemmatize
    return [stemmer.lemmatize(word) for word in nostop]

#Initialize tfidf vectorizer    
tfidf = TfidfVectorizer(analyzer=text_process)
#Fit text
print('Vectorizing Text....')
X = tfidf.fit_transform(df['Translated Text'])
#Tfidf matrix
X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names())

#Dimensionality Reduction 
seed = 42 #Set random seed for reproducability
np.random.seed(seed)
print('Reducing Dimensions....')
svd = TruncatedSVD(n_components=150, random_state=seed)
X_reduce = svd.fit_transform(X)


####################K-Means Clustering
#Employ Elbow Method to estimate possible k
print('Fitting K-Means....')
model = KMeans(random_state=seed)
visualizer = KElbowVisualizer(model, k= (5,20), metric='silhouette')

visualizer.fit(X_reduce) # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

##################Analysis of Chosen K
chosen = visualizer.elbow_value_ #Pit of Elbow (Chosen k)
kmeans = KMeans(n_clusters=chosen, random_state=seed)
kmeans.fit(X_reduce)
df['Cluster'] = kmeans.labels_

#Check articles of different langauges
topics = np.unique(df['Original Title'])
mismatch = [] #Store mismatched topics

for topic in topics:
    topic_df = df.loc[df['Original Title'] == topic]
    #Mismatch if more ore than 1 unique cluster per topic:
    if len(np.unique(topic_df['Cluster'])) > 1:
        mismatch.append(topic)
        #print(topic_df[['Original Title','Cluster']])

print()     
print('Total Mismatch Count: {}'.format(len(mismatch)))
print('Mismatched Articles: {}'.format(mismatch))
print()

#Cluster Contents
clusters = {}
for i in range(chosen):
    clusters[i] = []

for topic in topics:
    topic_df = df.loc[df['Original Title'] == topic]
    cluster_pred = topic_df['Cluster']
    
    #No Majority Cluster, Choose English
    if len(np.unique(cluster_pred)) == 3:
        eng_cluster = int(topic_df.loc[df['Detected Language'] == 'en','Cluster'])
        #Append topic to english cluster
        clusters[eng_cluster].append(topic)
        continue
    
    #Append Majority Cluster
    maj_cluster = max(list(cluster_pred), key=list(cluster_pred).count)
    clusters[maj_cluster].append(topic)
    
#Print cluster contents
for cluster in clusters:
    print('Cluster {}: {}'.format(cluster,clusters[cluster]))
print()        

#Cluster Visualizations
from yellowbrick.cluster import SilhouetteVisualizer
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')

visualizer.fit(X_reduce) # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(kmeans)

visualizer.fit(X_reduce) # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

#Export results to local storage
#df.to_csv('./files/text_final.csv', index=False, encoding='utf_8_sig')




