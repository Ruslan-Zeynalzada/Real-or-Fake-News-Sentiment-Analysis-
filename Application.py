import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
import random as rd

header = st.container()
data = st.container()
text = st.container()
template = st.container()
predictions = st.container()

with header : 
    st.title("Fake or Real News")
    st.markdown("* First you have to enter **any news** to the text box")
    st.markdown("* Then you can click **Predict** button and see the results")
    st.markdown("* If you don't want to write news you can select one randomly from **News Template** section on the left")


        


with data :
    st.header("The Dataset")
    df = pd.read_csv("Data_Set")
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    st.write(df.head())
    
    X = df["text"]
    y = df["label"]
   
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)
    
    def count(text) : 
        count = Counter()
        for x in text : 
            for i in x.split() : 
                count[i] += 1
        return count   
    Counter = count(text = df["text"])
    vocab_size = len(Counter)
    avg_sequence_size = round(np.mean([len(x.split()) for x in X_train]))
    
    token = Tokenizer(num_words=vocab_size , oov_token="OOV")
    token.fit_on_texts(X_train)
    
    
with text : 
    st.header("Giving News to the model")
    inputs = st.text_input("Enter News")

    
with template : 
    
    fake_news = ["Watch episode-5 Of Fact Vs Fiction where Suyash Barve & Jency Jacob discuss the fake news stories in the last one week and how you can spot them on social media.In this Video we cover the following stories:Modi Vs Gandhi: Twitter Analytics Reveal Fake Followers Plague Both Handles Twitter Fact Checks Kiran Bedi’s Fake Video Of PM Modi’s Mother Celebrating Diwali The Story Behind This Viral Image: Photo Fact FileReal Vs Reel: FactChecking Mersal’s Claims On GST, Demonetisation" , 
'The Bengaluru City Police’s official Twitter handle on Monday rubbished a hoax message that is doing the rounds about terrorists, involved in the serial Sri Lanka blasts, being spotted in the city early last week.\n\nIn the tweet, accompanied by screenshots of the viral set of messages, the police has urged people not fall for or forward such the message.\n\nThe tweet can be viewed below\n\nBOOM had also received the viral message on its WhatsApp helpline number (7700906111) with a caption claiming they are suspects in the recent Easter Sunday blasts in Sri Lanka who were now spotted in Bengaluru.\n\nThe messages mention an alleged spotting of the terrorists involved in the serial Sri Lanka blasts on Easter Sunday, in Bengaluru.\n\n“These are pics of Shrilanka blast suspects that Bangalore police who visited our apartment today have given us. They are looking for them as they are expected to be in Bangalore. The Police told us that these four people have entered Bangalore as seen in airport cctvs. And they maybe planning an attack in any of the IT parks. So we should lookout for people who come for interviews and for renting apartment. Our s being somewhat on the outskirts,so they came down. Yes we can share the information,” (sic.) read the message.\n\nAnother Whatsapp forward features a collage of five people, allegedly identified as the Sri Lanka blasts accused and warns citizens to be safe as they “plan to hit the city with a similar attack,” in the city as well.',
'A set of images is being shared on Facebook with the claim that it shows instances from the recent mob attack on West Bengal police during the COVID-19 lockdown at Howrah.\n\nThe claim is false; the photos are from June 2014, when Bharatiya Janata Party activists clashed with Uttar Pradesh police in front of the UP Assembly, while protesting against rising crimes in the state.\n\nBOOM spotted a post on Facebook by a user who shared the images with the caption: "Jai Hind, Jai bangla. Look at this. Howra Tikiapara. The picture speaks for itself - who\'s breaking lockdown to beat the police (translated from Bengali)." One of the images showed a person wearing a t-shirt with an image of Narendra Modi.\n\n\n\n\n\n\n\nThe post is being shared in the backdrop of a mob attack on West Bengal police, that took place on April 28 when the cops tried to enforce lockdown in Howrah, which has been identified as a red zone for COVID-19.\n\nFact Check\n\n\n\nBOOM ran a reverse image search on the photos, and came across a news report by The Telegraph from July 1, 2014, which carried one of the images. According to the caption, the image showed a BJP worker fight with a policeman during violent protests by the BJP against Akhilesh Yadav\'s government on June 30, 2014, for rising crime rates in the state.\n\n\n\n\n\n\n\nWhile the article did not carry the other image in the viral post, we found it being shared on image sharing website Imgur on July 2, 2014 - 2 days after the clash took place at the UP Assembly.\n\n\n\n\n\nWe compared the structure of the building seen in the photo with a more recent image of the UP Assembly building, and found it to be match. The date of the post and the resemblance of the venue helps us ascertain that this image is also from the clash between BJP workers and UP Police from June 2014.',
'While addressing the Lok Sabha on February 6, 2020, Prime Minister Narendra Modi quoted an article from satire website Faking News to attribute a fake quote to Omar Abdullah on the abrogation of Article 370. According to Modi, Abdullah had said that the abrogation of Article 370 will bring an earthquake, that will separate Kashmir from India. But the website Faking News clearly states that it is a satirical website. Read more details about the quote here.\n\n\n\n\n\nPosts on social media claiming that Chinese President Xi Jinping has been visiting mosques to offer prayers to protect China from the deadly Coronavirus outbreak have gone viral. But BOOM found that the photos in the posts show Jinping at a mosque are originally from his July 2016 visit to Xincheng mosque in Yinchuan city, and are not connected to the recent Coronavirus outbreak. Click here to read more details about his 2016 visit.\n\n\n\n\n\nDelhi assembly elections were held on February 8 but there was a lot of fake news that went viral just before that. A picture of CM Arvind Kejriwal posing with Pakistan\'s Prime Minister went viral on social media claiming to state that the two met ahead of the Assembly elections in Delhi. But BOOM\'s investigation found that the image was taken on May 2016 when Imran Khan visited Arvind Kejriwal in Delhi. Read the detailed story here.\n\n\n\n\n\nA viral photo showing the label of a Dettol disinfectant product mentioning its effectiveness in fighting the Coronavirus has led to confusion and misinformation online. Several netizens have posted the picture online and questioned how the company knew about the existence of the virus months before an outbreak was reported in December 2019. BOOM found the claim to be false as Coronavirus is the term used to describe a family of viruses that cause infections in both mammals and humans and does not refer to the recent novel Coronavirus outbreak. Read all the details about the term here.\n\n\n\n\n\nAn image of a fake screen grab of a news bulletin is doing the rounds on social media with claims that marijuana is effective in killing the deadly Coronavirus. BOOM did a reverse image search and found that the screen grab of what appears to be a news bulletin is from multiple meme pages carrying the same image as a "popular meme". Moreover, there is WHO has stated that there is no specific treatment available to treat the 2019-nCov as of now. Those infected get treatment and supportive care for the symptoms they show. Read more details about this meme and how it was created here.',
'A photograph of Defence Minister Nirmala Sitharaman standing along side a female army officer is being shared on Facebook with the claim that the woman in uniform is the defence minister’s daughter.\n\nHowever, BOOM was able to ascertain that the claim is false. Find out about the identity of the woman officer in our story here.\n\nA video claiming to be an appeal made by Prime Minister Narendra Modi’s estranged wife Jashodaben, asking people not to vote for Modi, has gone viral on the social media. However, a fact-check revealed that\n\nJashodaben, instead, is speaking about a statement reportedly given by the present Madhya Pradesh governor Anandiben Patel on the marital status of Prime Minister Narendra Modi. Read the story here.\n\n\n\n\n\nA Facebook post claiming Prime Minister Narendra Modi did not know how to respond to a question from a Muslim woman about his estranged wife Jashodaben, is false. The image is from an unrelated event and is being shared with a false context. Read the whole story here.\n\n\n\n\n\nA disturbing photo of a priest hanging from a temple wall in Uttar Pradesh is being shared on social media with a false narrative that Muslims killed the Hindu priest. But local police told BOOM that the FIR has been registered against four Hindus and there is no communal angle to the incident. Read more about it here.\n\nAn old photo of two women dressed in military uniform is being shared on Facebook with text in Bengali that claims they are female soldiers of the Indian Army at the Indo-Pak border. However, the women in the photograph appear to be wearing a uniform similar to that worn by Kurdish Peshmerga female fighters. Read the story here.']
    
    real = ["Kareena Kapoor Khan, who is all set to ring in her 40birthday tomorrow on September 21, took to her Instagram handle to share a monochrome selfie, smiling as she looks back to the milestones she achieved before entering her 40th year.Check out her post here:She wrote, As I enter my 40th year... I want to sit back, reflect, love, laugh, forgive, forget and most importantly pray and thank the strongest force up there for giving me the strength and thank my experiences and decisions for making me the woman I am... Some right, some wrong, some great, some not so... but still, hey BIG 40 make it BIG.The star icon is an epitome of expressions and has gifted her fans a filmography of nearly 60 films. She is currently expecting her second child with Saif Ali Khan . The couple had made an official statement last month saying, “We are very pleased to announce that we are expecting an addition to our family !!Thank you to all our well wishers for all their love and support.”On the work front, Kareena will next be seen in Karan Johar ’s period drama ‘Takht’ and Aamir Khan starrer, ‘Laal Singh Chaddha’." , 
'Advocate Ishkaran Bhandari recently linked Sushant Singh Rajput and Disha Salian ’s case. According to him, Siddharth Pithani has reportedly stated that Sushant fainted and feared for his own life after Disha’s demise and that is something that cannot be buried under the carpet. Ishkaran told a news channel that Sushant’s flatmate Siddharth Pithani is one of the material witnesses in the case who brought the late actor’s body down. According to Siddharth, Sushant fainted and later feared for his own life after Disha Salian’s death. This is something that cannot be swept under the carpet.He also added that he has been insisting right from the start that both these cases need to be investigated by the Central Bureau of Investigation (CBI). He also questioned why nobody from the June 8 party not coming out and giving statements. He also stated that no pictures or videos have come out from the night of the party.Like BJP MLA Nitesh Rane, Ishkaran also questioned Disha’s boyfriend Rohan Rai ’s absence. He also added that Rohan apparently came down 25 minutes after Disha’s body hit the ground.Disha Salian allegedly jumped from a high rise in Mumbai on June 8 and days later on June 14, Sushant Singh Rajput was found dead in his Mumbai apartment.',
'Sep 20, 2020, 08:00AM IST Source: TOI.in Meet Neelkantha Bhanu Prakash - the world’s fastest human calculator. The 21-year-old won India’s first gold medal in the Mental Calculation World Championship at Mind Sports Olympiad (MSO) held in London. He holds 4 world records and 50 Limca records for being the fastest human calculator in the world. TOI asked him some Math questions and this is how fast he solved them. Watch this video.',
'19:17 (IST) Sep 20\n\nThe second round of countrywide serosurvey led by ICMR has been successfully completed. The final phase analysis of the survey is now underway and will offer a comparison with the results of the first survey: Indian Council of Medical Research (ICMR)',
'Read Also\n\nRead Also\n\nAfter accusing Anurag Kashyap of sexually harassing her back in 2014, actress Payal Ghosh has now decided to file a police complaint against the filmmaker at Oshiwara Police Station, Mumbai on September 21. Payal’s lawyer Advocate Nitin Satpute released an official statement where he revealed that the actress was molested and was treated badly at Kashyap’s house. He added that the paperwork is still on and the actress will be filing an official police complaint on September 21.Elaborating further, the lawyer added that the actress tried to file a complaint before but she was apparently threatened. He also stated that she was pressurized that if she files a complaint, she would be boycotted. On Saturday, Payal Ghosh accused Anurag Kashyap of sexually harassing her and treating her badly at his house. The filmmaker, on the other hand, rubbished all her claims.While Payal found support in Kangana Ranaut, Kashyap was supported by his first wife Aarti Bajaj and his other Bollywood friends and counterparts.Payal also recently expressed her gratitude towards Kangana Ranaut for supporting her. Talking to her Twitter handle, she wrote, ‘Thank you so much for your support @KanganaTeam . This was high time and your support means a lot. We are women and we can together bring all of them down’Check out the tweet here:']
    st.sidebar.title("News Template")
  
    choice = st.sidebar.radio("Wich type of news do you want to select?" , ("Fake" , "Real"))

    if choice == "Fake" : 
        st.sidebar.header("You have selected Fake news. We selected one for you randomly")
        btn_3 = st.sidebar.button("Show news")
        btn_4 = st.sidebar.button("Predict choosen news")
        random_choice_fake = rd.choice(fake_news)
        if btn_3 :
            st.sidebar.header("The **Fake** News")
            st.sidebar.write(random_choice_fake)
        elif btn_4 : 
            model = tf.keras.models.load_model("NLP.h5")
            prediction = np.average(tf.math.round(tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(random_choice_fake) ,          maxlen=avg_sequence_size,padding="post" , truncating="post")))))
            proba = tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(random_choice_fake) ,maxlen=avg_sequence_size,padding="post" ,truncating="post")))
            st.sidebar.write("The prediction : **This news is Fake** and the probablity is {:.2%}".format(np.round(prediction), np.round(np.average(proba), 2)))
            
    elif choice == "Real" : 
        st.sidebar.header("You have selected Real news. We selected one for you randomly")
        random_choice_real = rd.choice(real)
        btn_5 = st.sidebar.button("Show News")
        btn_6 = st.sidebar.button("Predict choosen News")
        if btn_5 : 
            st.sidebar.write("The **Real** News")
            st.sidebar.write(random_choice_real)
        elif btn_6 : 
            model = tf.keras.models.load_model("NLP.h5")
            prediction = np.average(tf.math.round(tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(random_choice_real) ,          maxlen=avg_sequence_size,padding="post" , truncating="post")))))
            proba = tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(random_choice_real) ,maxlen=avg_sequence_size,padding="post" ,truncating="post")))
            st.sidebar.write("The prediction : **This news is Real** and the probablity is {:.2%}".format(np.round(prediction), np.round(np.average(proba), 2)))


            
with predictions :        
    btn = st.button("Predict")
    if btn : 
        model = tf.keras.models.load_model("NLP.h5")
        prediction = np.average(tf.math.round(tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(inputs) ,        maxlen=avg_sequence_size,padding="post" , truncating="post")))))
        proba = tf.squeeze(model.predict(pad_sequences(sequences=token.texts_to_sequences(inputs) ,        maxlen=avg_sequence_size,padding="post" , truncating="post")))
        st.write("The prediction is {} and the probablity is {:.2%}".format(np.round(prediction), np.round(np.average(proba), 2)))


