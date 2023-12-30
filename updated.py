import PyPDF2
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from difflib import ndiff
import nltk
import difflib

from pdf2image import convert_from_path
import pytesseract

try:
    
    nltk.data.find('corpora/stopwords.zip')
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    
    nltk.download('stopwords')
    nltk.download('punkt')

# Function to extract images from a PDF file
def extract_images_from_pdf(file_path):
    images = convert_from_path(file_path)
    return images

# Function to perform OCR on extracted images
def perform_ocr(images):
    ocr_text = ""
    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')  # Perform OCR on the image
        ocr_text += f"Text from Image {idx + 1}:\n{text}\n\n"
    return ocr_text


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
    diff = difflib.ndiff(text_1.splitlines(), text_2.splitlines())
    differences = [(i + 1, line) for i, line in enumerate(diff) if line.startswith('+ ') or line.startswith('- ')]
    return differences

input_pdf_path = 'modify.pdf'
original_pdf_path = 'org1.pdf'

uploaded_text = extract_text_pdfplumber(input_pdf_path)
original_text = extract_text_pdfplumber(original_pdf_path)

uploaded_tokens = preprocess_text(uploaded_text)
original_tokens = preprocess_text(original_text)

similarity_score = calculate_similarity_score(uploaded_tokens, original_tokens)

differences = find_differences(uploaded_text, original_text)
errors_count = len(differences)

if differences:
    print(f"Total Number of Errors: {errors_count}")
    print("Differences found between uploaded PDF and original PDF:")
    for line_num, line in differences:
        print(f"Error in line {line_num}: {line.replace('+ ', 'ORIGINAL ').replace('- ', 'EDITED ')}")
else:
    print("No differences found.")

print(f"\nSimilarity score: {similarity_score}")

# Generate HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDF Comparison Report</title>
    <style>
        .error {{
            background-color: red;
        }}
        .original-text {{
            color: blue;
        }}
        .edited-text {{
            color: green;
        }}
    </style>
</head>
<body>
    <h1>PDF Comparison Report</h1>
    <p>Total Number of Errors: {errors_count}</p>
    <h2>Differences Found:</h2>
    <ul>
"""

for line_num, line in differences:
    html_report += f"    <li>Error in line {line_num}: {line}</li>\n"

html_report += f"""
    </ul>
    <h2>Text Comparison:</h2>
    <h3>Original Text</h3>
    <p class="original-text">{original_text}</p>
    <h3>Edited Text</h3>
    <p class="edited-text">{uploaded_text}</p>
</body>
</html>
"""

# Save HTML report to a file with UTF-8 encoding
with open('comparison_report.html', 'w', encoding='utf-8') as file:
    file.write(html_report)
