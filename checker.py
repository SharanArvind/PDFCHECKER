from flask import Flask, render_template, request
import PyPDF2
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from difflib import ndiff
import nltk

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

def extract_text_pypdf2(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def extract_text_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w.lower()) for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return set(tokens)

def calculate_similarity_score(tokens_1, tokens_2):
    common_tokens = tokens_1.intersection(tokens_2)
    similarity_score = len(common_tokens) / len(tokens_2)
    return similarity_score

def find_differences(text_1, text_2):
    diff = list(ndiff(text_1.splitlines(), text_2.splitlines()))
    differences = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
    return differences

@app.route('/')
def index():
    return render_template('index.html', error=None, similarity_score=None)

@app.route('/process', methods=['POST'])
def process_pdf():
    original_pdf = request.files['originalPDF']
    edited_pdf = request.files['editedPDF']

    original_text = extract_text_pdfplumber(original_pdf)
    edited_text = extract_text_pdfplumber(edited_pdf)

    original_tokens = preprocess_text(original_text)
    edited_tokens = preprocess_text(edited_text)

    similarity_score = calculate_similarity_score(original_tokens, edited_tokens)
    differences = find_differences(original_text, edited_text)

    return render_template('index.html', error=differences, similarity_score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
