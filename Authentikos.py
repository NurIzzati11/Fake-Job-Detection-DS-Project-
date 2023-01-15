import pandas as pd
import joblib
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions
from PIL import Image

image_ok = Image.open("ok.png")
image_no = Image.open("notgood (1).png")
logo=Image.open("logo.png")
presentation=Image.open("presentation pic.PNG")
#logo=logo.resize((150,100))

model = joblib.load("model/tfidf_LR.pkl")

nlp = spacy.load("en_core_web_sm")

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: #expand contractions
        text = expand_contractions(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()

    doc = nlp(text) #tokenise text

    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)        
    return clean_text

def show_result(text):
    result=""
    dftest=pd.DataFrame()
    dftest['text']=[text]
    dftest['text']=dftest.apply(lambda x: text_preprocessing(x['text']), axis=1)
    dftest['text']=dftest.apply(lambda x: " ".join(x['text']), axis=1)
    test_predict= model.predict(dftest['text'])
    if test_predict==0:
        result="Not Fraud"
    else:
        result="Fraud!"
    return result

#Streamlit
import streamlit as st
co1, co2, co3 = st.columns(3)
with co1:
    st.write(' ')



with co2:
    #st.title("Authentikos")
    st.image(logo, use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Authentikos</h1>", unsafe_allow_html=True)
    
with co3:
    st.write(' ')

st.markdown("<h5 style='text-align: center; color: white;'>Is that job advertisement real or fake?</h5>", unsafe_allow_html=True)
st.write(" ")
tabs= st.tabs(["About","User Manual", "Fake Job Detector", "Project Information"])


tab_intro = tabs[0]
tab_intro.subheader("Introduction")
tab_intro.write("""
                Authentikos is a play of words from "Authenticate" and "Authentication" as this tool is to detect or authenticate
                the validity of a job advertisement received by users by classifying it either being **Fraud** or **Not Fraud**. 
                It is a web application built was on top of Python with Streamlit integration with its objective being to help job seekers know if the job
                advertisement they received is authenthic or not. 
                This is a Data Science Project for my 3rd year and I hope this tool will benefit its users.
                """)
tab_intro.subheader("Contact Me")
tab_intro.write("Feel free to contact me at:\n"
                "- Email  - izzatishal12@gmail.com\n"
                "Want to see the source code? Visit my repo in Github!\n"
                "- https://github.com/NurIzzati11/Fake-Job-Detection-DS-Project-"
                )

tab_manual = tabs[1]
tab_manual.subheader("User Manual")
tab_manual.write("**How to use this tool?**\n"
                 "1. Click the \"Fake Job Detector\" tab to arrive at the detector page\n"
                 "2. Enter the job advertisement text. You can enter everything from its title, benefits, requirements, etc.\n"
                 "3. Click \"Detect Now\" button and wait for your results.\n"
                )
tab_manual.caption("Note: Remember not to entirely rely on this tool. Your own judgement is also crucial. If in doubt, contact appropriate authorities for help.' ")


tab_detector= tabs[2]
tab_detector.subheader("Fake Job Detector")
tab_detector.write("""
                    How do you know if the job offer is fraudulent or real?
                    Try using this tool to find out!
                   """)
tab_detector.caption("_Note: Text must be **ONLY** in **:red[ENGLISH]**_")
user_input = tab_detector.text_area(label='Enter the job description')
if tab_detector.button('Detect Now'):
        answer = show_result(user_input)
        if answer == "Fraud!":           
            col1, col2, col3 = tab_detector.columns(3)
            with col1:
                tab_detector.write(' ')

            with col2:
                st.write(" ")
                st.markdown("<h2 style='text-align: center; color: white;'>Fraud!</h2>", unsafe_allow_html=True)
                st.image(image_no)
                st.markdown('<div style="text-align: center;">Oh no! This job prospect seems to good to be true!</div>', unsafe_allow_html=True)

            with col3:
                tab_detector.write(' ')
            #st.caption("Note: Remember not to entirely rely on this tool. Your own judgement is also crucial. If in doubt, contact appropriate authorities for help.' ")  
            
        else:
            col1, col2, col3 = tab_detector.columns(3)
            with col1:
                tab_detector.write(' ')

            with col2:
                st.write(" ")
                st.markdown("<h2 style='text-align: center; color: white;'>Not Fraud!</h2>", unsafe_allow_html=True)
                st.image(image_ok)
                st.markdown('<div style="text-align: center;">Even though it is not fraud, you still have to be careful!</div>', unsafe_allow_html=True)
                
                
            with col3:
                tab_detector.write(' ')
            #st.caption("Note: Remember not to entirely rely on this tool. Your own judgement is also crucial. If in doubt, contact appropriate authorities for help.' ") 


tab_project= tabs[3]
tab_project.subheader("Project Information")
tab_project.write("""
                  If you want to know more about the project (dataset, methodology, etc), please feel free to read the slides in the link below.
                  \nhttps://www.canva.com/design/DAFWf_MOSiU/gCNeDKpta5Q6UOuAoenOEA/view?utm_content=DAFWf_MOSiU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
                  """)
tab_project.image(presentation)
tab_project.write("")
