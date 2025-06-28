import streamlit as st
import pickle
import nltk
import re
import PyPDF2

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
model = pickle.load(open('models/model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))

# Keyword dictionary for section extraction
SECTION_KEYWORDS = {
    "education": ["education", "qualification", "academic"],
    "work_experience": ["experience", "employment", "work history"],
    "skills": ["skills", "technologies", "technical skills"],
    "certifications": ["certifications", "courses", "licenses"],
}


# Clean resume text
def clean_resume(resume_txt):
    clean_Txt = re.sub('http\S+\s', '', resume_txt)
    clean_Txt = re.sub('RT|cc', ' ', clean_Txt)
    clean_Txt = re.sub('#\S+', ' ', clean_Txt)
    clean_Txt = re.sub('@\S+', ' ', clean_Txt)
    clean_Txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_Txt)
    clean_Txt = re.sub(r'[^\x00-\x7f]', ' ', clean_Txt)
    clean_Txt = re.sub('\s+', ' ', clean_Txt)
    return clean_Txt


# Extract name (first line with 2–4 words, no digits)
def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        if len(line.split()) <= 4 and not any(char.isdigit() for char in line):
            return line.strip()
    return "Name not found"


# Extract sections based on keywords
def extract_sections(text):
    sections = {
        "education": "", "work_experience": "",
        "skills": "", "certifications": ""
    }
    lines = text.split('\n')
    current_section = None

    for line in lines:
        clean_line = line.strip().lower()
        for key, keywords in SECTION_KEYWORDS.items():
            if any(k in clean_line for k in keywords):
                current_section = key
                break
        else:
            if current_section and line.strip():
                sections[current_section] += line + '\n'

    return sections


# Read PDF file text
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text


# Main function
def main():
    st.title("AI Resume Screening App")
    upload_file = st.file_uploader('Upload Resume (.txt or .pdf)', type=['txt', 'pdf'])

    if upload_file is not None:
        if upload_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(upload_file)
        else:
            resume_bytes = upload_file.read()
            try:
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        # Clean and predict
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = model.predict(input_features)[0]

        category_mapping = {
            6: "Data Science", 12: "HR", 0: "Advocate", 1: "Arts",
            24: "Web Designing", 16: "Mechanical Engineer", 22: "Sales",
            14: "Health and fitness", 5: "Civil Engineer", 15: "Java Developer",
            4: "Business Analyst", 21: "SAP Developer", 2: "Automation Testing",
            11: "Electrical Engineering", 18: "Operations Manager",
            20: "Python Developer", 17: "DevOps Engineer",
            19: "Network Security Engineer", 7: "PMO", 13: "Database",
            10: "Hadoop", 9: "ETL Developer", 3: "DotNet Developer",
            23: "Blockchain",
        }

        category_name = category_mapping.get(prediction_id, 'Unknown')

        # Extract resume info
        name = extract_name(resume_text)
        sections = extract_sections(resume_text)

        # Output
        st.subheader("Predicted Resume Category:")
        st.success(category_name)

        st.subheader("Extracted Resume Information:")
        st.write("**Name:**", name)
        st.write("**Education:**", sections['education'] or "Not found")
        st.write("**Work Experience:**", sections['work_experience'] or "Not found")
        st.write("**Skills:**", sections['skills'] or "Not found")
        st.write("**Certifications:**", sections['certifications'] or "Not found")


if __name__ == "__main__":
    main()






