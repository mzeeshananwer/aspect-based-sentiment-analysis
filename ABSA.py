#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[26]:


import pandas as pd
import numpy as np
from datetime import datetime
import locale
import re
from tqdm import tqdm
from transformers import pipeline
import os
from langdetect import detect
import math
import argparse
import random
import pickle
import string
import cufflinks as cf
import chart_studio.plotly as py
import emoji
from emot.emo_unicode import EMOTICONS_EMO, UNICODE_EMOJI_ALIAS
from plotly import __version__
get_ipython().run_line_magic('matplotlib', 'inline')
#import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
#cf.go_offline()
tqdm.pandas()


# In[4]:


import spacy
nlp = spacy.load("en_core_web_sm")


# In[28]:


### Required modules
#!pip install emot
# Installing required libraries
#!pip install SoMaJo
# pycontractions better but difficult to install
#!pip install --upgrade contractions
#!pip install contextualSpellCheck
#!pip install deep-translator 
#!pip install -U spacy & python -m spacy download en_core_web_sm
#!pip install emoji
#!pip3 install -U pyabsa==2.0.2
#!pip install cufflinks
#!pip install chart_studio


# In[6]:


raw_data = pd.read_csv(r"01_Amazon_raw.csv",
                        sep = "|",
                        decimal = ",",
                        encoding="utf-8-sig")


# In[7]:


def describe_full(df):
    dtypes_description=pd.DataFrame(dict(df.dtypes),["dtypes"])
    na_description = pd.DataFrame(dict(df.isna().sum()),["NA-s"])
    na_percent = ((pd.DataFrame(dict(df.isna().sum()),["NA%"])/len(df))*100).round(decimals=2)
    description = df.describe(include="all")
    full_description = dtypes_description.append(na_description).append(na_percent).append(description).replace(np.nan, "", regex=True)

    mask = full_description.loc["freq",:]==1
    full_description.at[["top"],mask.index[mask]]=""
    
    return full_description

describe_full(raw_data)


# We learn that there are 58/60 unique products from 11 different companies

# In[8]:


print(raw_data['product'].nunique())
print(raw_data['product'].unique())


# In[9]:


raw_data.country.unique()


# In[10]:



raw_data["title_and_text"] = raw_data["title"] + raw_data["text"]

def get_language(text):
    try:
        language = detect(text)
    except :
        language = "N/A"
    return language

print("Start language detection")

tqdm.pandas()
raw_data["review_language"] = raw_data["title_and_text"].progress_apply(lambda text: get_language(text))

print("Language detection done")


# In[11]:


raw_data["review_language"].unique()


# In[12]:


raw_data.to_pickle('data_with_lang.pkl')


# In[13]:


english_reviews = raw_data.loc[raw_data["review_language"]=="en"]
english_reviews.to_pickle("./eng_reviews.pkl")


# In[14]:


english_reviews.shape


# In[15]:


english_reviews


# There are 171,151 English reviews and 78046 German reviews

# In[16]:


data = pd.read_pickle('data_with_lang.pkl')
#data.to_pickle('data_with_lang3.pkl', protocol=3)


# In[17]:


data.shape


# In[18]:


data.columns


# In[19]:


data.review_language.unique()


# In[20]:


print(data.review_language.nunique())


# In[21]:


data.groupby('review_language').describe()


# In[22]:


data.groupby('company').describe()


# In[23]:


ratings_en = {"1.0 out of 5 stars": 1.0,
            "2.0 out of 5 stars": 2.0,
            "3.0 out of 5 stars": 3.0,
            "4.0 out of 5 stars": 4.0,
            "5.0 out of 5 stars": 5.0}

ratings_fr = {"1,0 sur 5\xa0étoiles": 1.0,
            "2,0 sur 5\xa0étoiles": 2.0,
            "3,0 sur 5\xa0étoiles": 3.0,
            "4,0 sur 5\xa0étoiles": 4.0,
            "5,0 sur 5\xa0étoiles": 5.0}


# In[24]:


data = data.replace({"rating": ratings_fr})
data = data.replace({"rating": ratings_en})


# In[30]:


cf.set_config_file(offline=True, world_readable=True)
data['rating'].iplot(
    kind='hist',
    xTitle='rating',
    linecolor='black',
    yTitle='count',
    title='Review Rating Distribution')


# In[31]:


data['Positively Rated'] = np.where(data['rating'] > 3, 1, 0)
data['Positively Rated'].mean()


# In[32]:


data_with_no_comments = data[data['text'].str.len() == 0|data['text'].isnull()]
data_with_no_comments.shape


# In[33]:


data_with_no_comments['rating'].value_counts(normalize=True) * 100


# In[34]:


data['rating'].value_counts(normalize=True) * 100


# In[35]:


data.groupby('company')['rating'].describe()


# In[36]:


data.groupby('product')['rating'].describe()


# In[37]:


data['review_len'] = data['text'].str.split().str.len()
data.review_len.describe()


# In[38]:


cf.set_config_file(offline=True, world_readable=True)
data['review_len'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Text Length Distribution')


# In[39]:


y0 = data.loc[data['company'] == 'Bose']['rating']
y1 = data.loc[data['company'] == 'Sennheiser']['rating']
y2 = data.loc[data['company'] == 'JBL']['rating']
y3 = data.loc[data['company'] == 'AKG']['rating']
y4 = data.loc[data['company'] == 'Soundcore-Anker']['rating']
y5 = data.loc[data['company'] == 'Sony']['rating']

trace0 = go.Box(
    y=y0,
    name = 'Bose',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Sennheiser',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'JBL',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'AKG',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'Soundcore-Anker',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'Sony',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )
)
data_g = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title = "Sentiment Polarity Boxplot based on Company"
)

fig = go.Figure(data=data_g,layout=layout)
iplot(fig, filename = "Sentiment Polarity Boxplot based on Company")


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer


# In[42]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(data['text'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review before removing stop words')


# In[43]:


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(data['text'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df3.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[44]:


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(data['text'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df5.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[45]:


eng_data = data.loc[data.review_language=='en']


# In[46]:


def add_new_stopwords():
    # New stop words list 
    customize_stop_words = [
        'blah'
    ]
    # Mark them as stop words
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True
    
def remove_stops(doc):
    # Filter out stop words by using the `token.is_stop` attribute
    return ' '.join([token.text for token in doc if not token.is_stop])

def remove_stop_and_lemmatize(doc):
    # Take the `token.lemma_` of each non-stop word
    try: 
        return ' '.join(token.lemma_ for token in nlp(doc) if not token.is_stop)
    except Exception as e:
        print(doc)
        print(e)
        return ''

def clean(doc):
    if isinstance(doc, str):
        return remove_stop_and_lemmatize(doc)
    else:
        return ''


# In[47]:


eng_data['cleaned_review'] = eng_data['text'].map(lambda x: clean(x), na_action=None) 


# In[48]:


common_words = get_top_n_words(eng_data['cleaned_review'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review after removing stop words')


# In[49]:


common_words = get_top_n_bigram(eng_data['cleaned_review'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df3.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# In[50]:


common_words = get_top_n_trigram(eng_data['cleaned_review'].astype(str), 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
cf.set_config_file(offline=True, world_readable=True)
df5.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# In[51]:


eng_data.to_pickle('eng_lemma_stop.pkl')


# # Creating dataset for training ABSA

# In[52]:


# skip cleaning as lemmatization and stopword removal not important for contextual language models
english_reviews = data.loc[data.review_language=='en']
german_reviews = data.loc[data.review_language=='de']


# In[53]:


# remove contractions from English reviews
#using in-bult library contractions instead. The library pycontractions would perform better but difficult to install


import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

'''
if you want to add new contractions
contractions.add('mychange', 'my change')
'''


# In[63]:


def expand_contractions(doc):
    return contractions.fix(doc)

def perform_spell_check(text):
    #print(len(text._.suggestions_spellCheck)) # print number of errors detected
    #print(text._.suggestions_spellCheck) # print suggested spelling corrections
    try:
        nlp_text = nlp(text)
        #print(nlp_text._.suggestions_spellCheck)
        #print(nlp_text._.score_spellCheck)
        return nlp_text._.outcome_spellCheck
    except:
        return text
    
def convert_to_lower(text):
    return text.str.lower()

def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    ''' 
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    ''' 
    return BeautifulSoup(text, 'html.parser').get_text()

# Function for converting emojis and emoticons into text
def convert_emojis_to_word(text, method = 2):
    if method == 1:
        return emoji.demojize(text, language='en') # multiple languages supported
    
    elif method == 2:
        # replace emoticon :)
        for emoticon in EMOTICONS_EMO:
            text = text.replace(emoticon, "_".join((EMOTICONS_EMO[emoticon] + ' ').replace(",","").split()))
        # replace emoji
        for emoji in UNICODE_EMOJI_ALIAS:
            text = text.replace(emoji, (UNICODE_EMOJI_ALIAS[emoji] + ' ').replace(",","").replace(":",""))
        return text
    
    else:
        # 'Emoji_Dict.p'- download link https://drive.google.com/open?id=1G1vIkkbqPBYPKHcQ8qy0G2zkoab2Qv4v
        with open('Emoji_Dict.p', 'rb') as fp:
            Emoji_Dict = pickle.load(fp)
        
        Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

        for emot in Emoji_Dict:
            text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
        return text

def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    Warning! Always perform last
    '''
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocess_pipeline(text):
    return convert_emojis_to_word(remove_url_func(remove_extra_whitespaces_func(text)))


# In[64]:


example1 = 'Noice breaks when connected with laptop with Ubuntu18.04 installed. Works fine with windows 10.'
print(perform_spell_check(example1))

example2 = 'Small size is useful and they souns better than I expected'
print(perform_spell_check(example2))

example3 = 'Small size is useful and the ear phones souns better than I expected'
print(perform_spell_check(example3))


# In[65]:


example = u'The sound is excellent but they connect without knowing and the battery dies :(, also most of the time only one earbud is pairing not both of them! 👎👎👎'
convert_emojis_to_word(example)


# In[66]:


english_reviews['clean_text'] = english_reviews['text'].map(lambda text: preprocess_pipeline(text))


# In[67]:


english_reviews.to_csv('English_text.csv',sep='|', encoding="utf-8-sig")


# In[68]:


test_english_reviews = english_reviews.sample(200, random_state=40)
test_german_reviews = german_reviews.sample(200, random_state=40)


# In[69]:


train_english_reviews=english_reviews.drop(test_english_reviews.index)
train_german_reviews=german_reviews.drop(test_german_reviews.index)


# # Word Segmentation required for data annotation

# In[70]:


from somajo import SoMaJo

#language : {'de_CMC', 'en_PTB'} for tokenizing and segmenting German and English texts respectively

def tokenize(review, language):
    tokenizer = SoMaJo(language, split_camel_case=True)
    tokenized_reviews = tokenizer.tokenize_text(review)
    final_reviews = ''
    for review in tokenized_reviews:
        tokens = []
        for token in review:
            tokens.append(token.text)
        if final_reviews:
            final_reviews = final_reviews + ' ' + (' '.join(tokens))
        else:
            final_reviews = (' '.join(tokens))
        
    return final_reviews
    


# In[71]:


test_english_reviews['tokenized_clean_text'] = test_english_reviews['clean_text'].map(lambda x: tokenize(x.split(), 'en_PTB'))


# In[72]:


test_english_reviews['tokenized_clean_text'].head


# In[73]:


test_english_reviews['tokenized_clean_text'].to_csv('test_eng.csv', index=False, header=False)


# In[74]:


test_german_reviews['tokenized_clean_text'] = test_german_reviews['text'].map(lambda x: tokenize(x.split(), 'de_CMC'))
test_german_reviews['tokenized_clean_text'].to_csv('test_german.csv', index=False, header=False)


# # Other experiments

# In[75]:


from deep_translator import GoogleTranslator
'''
possible translators = GoogleTranslator,api_key = 'd2287fda8e218154ff47bf04552adb42' #free plan: 1,000 requests/day or 1 MB/day
                       MicrosoftTranslator,
                       PonsTranslator,
                       LingueeTranslator,
                       MyMemoryTranslator,
                       YandexTranslator,
                       PapagoTranslator,
                       DeeplTranslator,
                       QcriTranslator,
                       single_detection,
                       batch_detection
Most require api_keys, have a free tier and pay-as-you-go pricing
'''


# In[76]:


translated_words = LingueeTranslator(source='german', target='english').translate_words(['Das klingt gut!', 'Gibt bessere für weniger Geld sogar .'])
translated_words


# In[77]:


translated_words = GoogleTranslator(source='german', target='english').translate('Gibt bessere für weniger Geld sogar.')
translated_words


# In[79]:


# Additional preprocessing Gibberish detection

#!pip install gibberish-detector
from gibberish_detector import detector
# load the gibberish detection model
Detector = Detector = detector.create_from_model('big.model')
text1 = "xdnfklskasqd"
print(Detector.is_gibberish(text1))
text2 = "apples"
print(Detector.is_gibberish(text2))


# # Evaluating PyABSA

# In[88]:


get_ipython().system('pip install -U pyabsa')


# In[89]:


get_ipython().system('pip show pyabsa')


# In[83]:


get_ipython().system('pip install --upgrade pyabsa')


# In[90]:


from pyabsa import available_checkpoints
from pyabsa import ABSADatasetList
from pyabsa import ATEPCCheckpointManager

checkpoint_map = available_checkpoints()
checkpoint_map


# In[95]:


aspect_multilingual_extractor = ATEPCCheckpointManager.get_aspect_extractor('multilingual')
#aspect_multilingual_extractor = ATEPCCheckpointManager.get_aspect_extractor('multilingual-256')
#aspect_multilingual_extractor = ATEPCCheckpointManager.get_aspect_extractor('multilingual-256-2')


# In[98]:


atepc_result = aspect_multilingual_extractor.extract_aspect(inference_source=test_german_reviews['text'].tolist(),
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )


# In[99]:


atepc_result = aspect_multilingual_extractor.extract_aspect(inference_source=test_english_reviews['text'].tolist(),
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )


# In[100]:


atepc_result = aspect_multilingual_extractor.extract_aspect(inference_source=test_english_reviews['clean_text'].tolist(),
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )


# In[101]:


aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor('english')
atepc_result = aspect_extractor.extract_aspect(inference_source=test_english_reviews['text'].tolist(),
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )


# In[109]:


atepc_result


# In[144]:


# Extract the 'aspect', 'sentiment', and 'confidence' values from atepc_result
aspects = [result['aspect'] for result in atepc_result]
sentiments = [result['sentiment'] for result in atepc_result]
confidences = [result['confidence'] for result in atepc_result]

# Create a dictionary with the extracted values
extracted_values = {'aspect': aspects, 'sentiment': sentiments, 'confidence': confidences}

# Convert the dictionary into a DataFrame
df_1 = pd.DataFrame(extracted_values)[1:]


# In[145]:


df_1


# In[146]:


new_aspect = []
new_sentiment = []
new_confidence = []

for index, row in df_1.iterrows():
    aspects = row['aspect']
    sentiments = row['sentiment']
    confidences = row['confidence']
    for aspect, sentiment, confidence in zip(aspects, sentiments, confidences):
        new_aspect.append(aspect)
        new_sentiment.append(sentiment)
        new_confidence.append(confidence)

new_df = pd.DataFrame({'aspect': new_aspect, 'sentiment': new_sentiment, 'confidence': new_confidence})


# In[147]:


new_df


# In[148]:


negative_df = new_df[new_df['sentiment'] == 'Negative']


# In[149]:


negative_df


# In[125]:


import matplotlib.pyplot as plt


plt.barh(negative_df['aspect'], negative_df['confidence'], height=1.5)
plt.xlabel('Aspect')
plt.ylabel('Confidence')
plt.title('Negative Sentiments')
#plt.xticks(rotation=90)
plt.show()


# In[130]:


import matplotlib.pyplot as plt

negative_df = new_df[new_df['sentiment'] == 'Negative']
negative_df = negative_df.sort_values(by='confidence', ascending=False).head(30)

plt.bar(negative_df['aspect'], negative_df['confidence'])
plt.xlabel('Aspect')
plt.ylabel('Confidence')
plt.title('Top 30 Negative Sentiments')
plt.xticks(rotation=90)
plt.show()


# In[131]:


# Make own dataset
from pyabsa import make_ABSA_dataset 

make_ABSA_dataset(dataset_name_or_path='review', checkpoint='english')


# In[132]:


# Fine-tune model on own dataset

import random

from pyabsa import AspectTermExtraction as ATEPC

config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
config.evaluate_begin = 0
config.max_seq_len = 512
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.l2reg = 1e-8
config.seed = random.randint(1, 100)
config.use_bert_spc = True
config.use_amp = False
config.cache_dataset = False

chinese_sets = ATEPC.ATEPCDatasetList.Multilingual

aspect_extractor = ATEPC.ATEPCTrainer(config=config,
                                      dataset=chinese_sets,
                                      checkpoint_save_mode=1,
                                      auto_device=True
                                      ).load_trained_model()


# In[133]:


from pyabsa import AspectTermExtraction as ATEPC
aspect_multilingual_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )


# In[135]:


atepc_result_m = aspect_multilingual_extractor.extract_aspect(inference_source=test_english_reviews['text'].tolist(),
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )


# In[136]:


# Extract the 'aspect', 'sentiment', and 'confidence' values from atepc_result
aspects = [result['aspect'] for result in atepc_result_m]
sentiments = [result['sentiment'] for result in atepc_result_m]
confidences = [result['confidence'] for result in atepc_result_m]

# Create a dictionary with the extracted values
extracted_values = {'aspect': aspects, 'sentiment': sentiments, 'confidence': confidences}

# Convert the dictionary into a DataFrame
df_m = pd.DataFrame(extracted_values)[1:]


# In[137]:


new_aspect = []
new_sentiment = []
new_confidence = []

for index, row in df_m.iterrows():
    aspects = row['aspect']
    sentiments = row['sentiment']
    confidences = row['confidence']
    for aspect, sentiment, confidence in zip(aspects, sentiments, confidences):
        new_aspect.append(aspect)
        new_sentiment.append(sentiment)
        new_confidence.append(confidence)

new_df_m = pd.DataFrame({'aspect': new_aspect, 'sentiment': new_sentiment, 'confidence': new_confidence})


# In[140]:


negative_df_m = new_df_m[new_df_m['sentiment'] == 'Negative']


# In[141]:


negative_df_m


# In[142]:


import matplotlib.pyplot as plt

negative_df_m = negative_df_m.sort_values(by='confidence', ascending=False).head(30)

plt.bar(negative_df['aspect'], negative_df['confidence'])
plt.xlabel('Aspect')
plt.ylabel('Confidence')
plt.title('Top 30 Negative Sentiments')
plt.xticks(rotation=90)
plt.show()


# In[151]:


import matplotlib.pyplot as plt

negative_df = new_df[new_df['sentiment'] == 'Negative']
negative_df = negative_df.sort_values(by='confidence', ascending=False).tail(30)

plt.bar(negative_df['aspect'], negative_df['confidence'])
plt.xlabel('Aspect')
plt.ylabel('Confidence')
plt.title('Top 30 Negative Sentiments')
plt.xticks(rotation=90)
plt.show()

