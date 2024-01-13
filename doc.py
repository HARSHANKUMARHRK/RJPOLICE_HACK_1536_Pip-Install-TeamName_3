import PyPDF2
import re
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def answer_question(context, question):
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased-whole-word-masking-finetuned-squad')
    result = qa_pipeline(context=context, question=question)
    return result['answer']

def main():
    pdf_path = 'law.pdf'
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(pdf_text)

    while True:
        user_question = input("Ask a question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        answer = answer_question(cleaned_text, user_question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
