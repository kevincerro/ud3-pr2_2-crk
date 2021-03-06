# from classifier import SentimentClassifier
from flask import Flask, render_template, request
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.language import Language
from spacy_langdetect import LanguageDetector

ENTITY_TYPES_LANG = {
    'en': {
        'PERSON': 'PERSON',
        'ORG': 'ORG',
        'PRODUCT': 'PRODUCT',
        'MONEY': 'MONEY',
        'DATE': 'DATE',
    },
    'es': {
        'PERSON': 'PER',
        'ORG': 'ORG',
        'PRODUCT': 'MISC',
        'MONEY': '',
        'DATE': '',
    }
}

app = Flask(__name__)

sid = SentimentIntensityAnalyzer()
# sc = SentimentClassifier()


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Get form data
    rawtext = request.form['rawtext']
    taskoption = request.form['taskoption']

    # Parse doc based on detected lang
    doc = get_doc_for_lang(rawtext)
    lang = doc._.language['language']

    if lang == 'es':
        # Convert to -1 / 1 range
        # sentiment = (sc.predict(rawtext) * 2) - 1
        # Disabled due spanish sentiment analisys is not compatible with heroku deps
        sentiment = 0
    else:
        sentiment = sid.polarity_scores(rawtext)['compound']

    # Prepare data for template
    results = doc.ents
    results_filtered = list(filter(lambda e: e.label_ == ENTITY_TYPES_LANG.get(lang, ENTITY_TYPES_LANG['en'])[taskoption], doc.ents))
    num_of_results = len(results_filtered)

    return render_template('index.html', results=results, results_filtered=results_filtered, num_of_results=num_of_results, lang=lang, sentiment=sentiment)


def get_doc_for_lang(text):
    en_nlp = spacy.load('en_core_web_md')
    en_nlp.add_pipe('language_detector', last=True)


    lang = en_nlp(text)._.language['language']
    if lang == 'es':
        es_nlp = spacy.load('es_core_news_md')

        return es_nlp(text)
    else:
        return en_nlp(text)
