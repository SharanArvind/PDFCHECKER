import PyPDF2
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from difflib import ndiff
import nltk

# Download stopwords resource
nltk.download('stopwords')
# Download punkt resource
nltk.download('punkt')

# Function to extract text from a PDF file using PyPDF2
def extract_text_pypdf2(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

# Function to extract text from a PDF file using pdfplumber
def extract_text_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to preprocess text: tokenize, remove stop words, and stem
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w.lower()) for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return set(tokens)  # Returning as a set for faster comparison

# Function to calculate similarity score based on token overlap
def calculate_similarity_score(tokens_1, tokens_2):
    common_tokens = tokens_1.intersection(tokens_2)
    similarity_score = len(common_tokens) / len(tokens_2)  # Using tokens from the original as the denominator
    return similarity_score

# Function to find differences and display them
def find_differences(text_1, text_2):
    diff = list(ndiff(text_1.splitlines(), text_2.splitlines()))
    differences = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
    return differences

# Example usage:
input_pdf_path = 'modify.pdf'
original_pdf_path = 'org1.pdf'

# Extract text from the uploaded PDF and the original PDF
uploaded_text = extract_text_pdfplumber(input_pdf_path)
original_text = extract_text_pdfplumber(original_pdf_path)

# Preprocess text for comparison
uploaded_tokens = preprocess_text(uploaded_text)
original_tokens = preprocess_text(original_text)

# Calculate similarity score
similarity_score = calculate_similarity_score(uploaded_tokens, original_tokens)

# Find and display differences
differences = find_differences(uploaded_text, original_text)
if differences:
    print("Differences found between uploaded PDF and original PDF:")
    for line in differences:
        print(line)
else:
    print("No differences found.")
    
# Display similarity score
print(f"\nSimilarity score: {similarity_score}")