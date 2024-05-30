import nltk
import re
import PyPDF2
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import DutchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import magic
import mimetypes
import langdetect
import torch
import transformers

class api_atm:

    __instance = None
    
    ##############################################################################
    #  
    #  __new__
    #
    #  Init the object and force a singleton
    #
    #  @param cls
    #  @param *args
    #  @param *kwargs
    #  @return object
    #
    ##############################################################################
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance
    

    ##############################################################################
    #  
    #  __init__
    #
    #  Init the object
    #
    #  @param void
    #  @return void
    #
    ##############################################################################
    def __init__(self):
        self.error = ''

        nltk.download('punkt')
        nltk.download('stopwords')


    ##############################################################################
    #  
    #  process_text
    #
    #  Process a text
    #
    #  @param string text
    #  @return string
    #
    ##############################################################################
    def process_text(self, text):
        return self._clean_dutch_text(text)


    ##############################################################################
    #  
    #  process_pdf
    #
    #  Process a PDF
    #
    #  @param string localFile
    #  @return json
    #
    ##############################################################################
    def process_pdf(self, localFile):

        # init result
        result = {"text"     : "",
                  "cleaned"  : "",
                  "mime"     : "",
                  "language" : "",
                  "title"    : "",
                  "keywords" : {},
                  "summary"  : "", 
                  "error"    : "",
                  "size"     : 0,
                  "id_list"  : {"kvk" : {}, "bsn" : {}},
                  "status"   : 500}

        # Get filesize
        result["size"] = os.path.getsize(localFile)

        # get MIME type
        mime           = magic.Magic()
        result["mime"] = mime.from_file(localFile)

        # First attempt to extract text
        text = self._extract_text_from_pdf(localFile)

        # Detect language
        result["language"] = langdetect.detect(text)

        # If empty string, return error
        if (len(text) == 0):
            result["error"] = self.error
            return result
        else:
            result["text"]     = text
            result["cleaned"] = self._clean_dutch_text(text)
            result["status"]   = 200

            # get working title
            workingTitle = self._generate_working_title(result["cleaned"])

            if (workingTitle == ""):
                result["error"] = self.error    
            else:
                result["title"] = workingTitle

            # Top keywords
            result["keywords"] = self._extract_top_keywords(result["cleaned"])
            
            # Summary
            result["summary"] = self._generate_summary(result["cleaned"])

            # KVK / BSN ID's
            result["id_list"]["bsn"] = self._findRegEx(r'\b(\d{9})\b', text)
            result["id_list"]["kvk"] = self._findRegEx(r'\b\d{8}\b', text)

        return result


    ##############################################################################
    #  
    #  _extract_text_from_pdf
    #
    #  Extract text from a PDF
    #
    #  @param string localFile
    #  @return json
    #
    ##############################################################################
    def _extract_text_from_pdf(self, pdfFile):
    
        text = ""

        try:
            with open(pdfFile, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            
                for page_num in range(len(pdf_reader.pages)):
                    page  = pdf_reader.pages[page_num]
                    text += page.extract_text()
        except Exception as ex:
            self.error = str(ex)

        return text
    

    ##############################################################################
    #  
    #  _clean_dutch_text
    #
    #  Clean a text using Dutch language options
    #
    #  @param string text
    #  @return string
    #
    ##############################################################################
    def _clean_dutch_text(self, text):
    
        # Remove email addresses and hyperlinks
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove content inside "<...>" and "{{...}}"
        text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', text)
        text = re.sub(r'{{[^>]+}}', '', text)

        # Remove non-alphanumeric characters and punctuation marks
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize text and remove stopwords
        words      = word_tokenize(text, language='dutch')
        stop_words = set(stopwords.words('dutch'))
        words      = [word for word in words if word.isalpha() and word not in stop_words]

        return ' '.join(words)
    

    ##############################################################################
    #  
    #  _generate_working_title
    #
    #  Create a working title from the cleaned text
    #
    #  @param string text
    #  @return string
    #
    ##############################################################################
    def _generate_working_title(self, text):

        title_keywords = ""

        try:
            vectorizer        = TfidfVectorizer()
            tfidf_matrix      = vectorizer.fit_transform([text])
            feature_names     = vectorizer.get_feature_names_out()
            tfidf_scores      = tfidf_matrix.toarray()[0]
            word_tfidf_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}
            title_keywords    = sorted(word_tfidf_scores, key=word_tfidf_scores.get, reverse=True)[:3]
            title_keywords    = ' '.join(word.capitalize() for word in title_keywords)
            
        except Exception as ex:
            self.error = str(ex)

        return title_keywords


    ##############################################################################
    #  
    #  _extract_top_keywords
    #
    #  Get top used keywords
    #
    #  @param string text
    #  @return array
    #
    ##############################################################################
    def _extract_top_keywords(self, text):

        vectorizer        = TfidfVectorizer()
        tfidf_matrix      = vectorizer.fit_transform([text])
        feature_names     = vectorizer.get_feature_names_out()
        tfidf_scores      = tfidf_matrix.toarray()[0]
        word_tfidf_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}

        return sorted(word_tfidf_scores, key=word_tfidf_scores.get, reverse=True)[:10]

    ##############################################################################
    #  
    #  _generate_summary
    #
    #  Create a summary from the cleaned text
    #
    #  @param string text
    #  @return string
    #
    ##############################################################################
    def _generate_summary(self, text):
        model = transformers.MBartForConditionalGeneration.from_pretrained("ml6team/mbart-large-cc25-cnn-dailymail-nl-finetune")
        tokenizer = transformers.MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

        pipeline_summarize = transformers.pipeline(
            task="summarization",
            model=model,
            tokenizer=tokenizer
        )

        pipeline_summarize.model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
            "nl_XX"
        ]
        # Encode the cleaned text with Dutch language code
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs, top_p=0.95, top_k=3, max_length=64,
                                    min_length=0, length_penalty=2.0,
                                    num_beams=2, early_stopping=True, do_sample=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
        return summary

    ##############################################################################
    #  
    #  _findRegEx
    #
    #  Find strings using a regex
    #
    #  @param string text
    #  @param string regex
    #  @return array
    #
    ##############################################################################
    def _findRegEx(self, pattern, text):
        return re.findall(pattern, text)
