import requests
from bs4 import BeautifulSoup
import wikipediaapi
import pandas as pd
from googletrans import Translator

####################Extract Top 50 Wikipedia Articles in 2019 list
source = requests.get("https://en.wikipedia.org/wiki/Wikipedia:2019_Top_50_Report").text
soup = BeautifulSoup(source,'lxml')

#Get table containing required content
table = soup.find("table", class_="wikitable")
rows = table.find_all("tr")
del rows[0] #Remove headers in table

#Grab all 50 topics
topics = [row.a.text for row in rows]
unwanted = ["List of highest-grossing films","Deaths in 2019","List of Bollywood films of 2019","2019 in film","Murder of Dee Dee Blanchard"]
#Remove unwanted Non-Articles
topics = [topic for topic in topics if topic not in unwanted]
#Include other articles instead
topics += ["Natural language processing","Data Science","Artificial intelligence","Machine learning","Singapore"]



###################Extract summary text of articles
languages = ["fr","zh"] #Langauges include English, French and Chinese
wiki = wikipediaapi.Wikipedia('en')

#Dataframe which will store the textual data (summary of articles)
df = pd.DataFrame(columns = ['Original Title','Language','Text'])

#Scrape and append
for topic in topics:
    print('Extracting Text for: {}'.format(topic))
    page_py = wiki.page(topic)
    df = df.append(pd.Series([topic,"en",page_py.summary], index = df.columns),ignore_index=True) 
    for lang in languages:
        page_py_lang = page_py.langlinks[lang]
        df = df.append(pd.Series([topic,lang,page_py_lang.summary], index = df.columns),ignore_index=True) 

#Export to csv for local storage
#df.to_csv('./files/text.csv', index=False, encoding='utf_8_sig')



######################Language Detection & Translation
#Initialise translator object
#translator = Translator()
translator = Translator(service_urls=[
        #'translate.google.com',
        'translate.google.co.kr',
        ])

all_translated = [] #Translated Texts
all_src = [] #Detected Languages

#Translate & Detect Language for all text
for i in range(df.shape[0]):
    #Print Translation Progress
    print("Translating for '{}' ({})".format(df.loc[i]['Original Title'],df.loc[i]['Language']))
    translated = translator.translate(df.loc[i]['Text'], dest='en')
    all_translated.append(translated.text)
    all_src.append(translated.src)
    
#Append Translations and Detected Langauges
df['Detected Language'] = all_src
df['Translated Text'] = all_translated

#Export to csv for local storage
#df.to_csv('./files/text_translated.csv', index=False, encoding='utf_8_sig')