import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("linear_svc_model.pkl")         # Your trained Linear SVC
vectorizer = joblib.load("tfidf_vectorizer.pkl")    # The TF-IDF vectorizer used in training

# Define simple criteria for Accept/Reject
REQUIRED_SKILLS = ['python', 'data', 'machine learning', 'sql']  # example
MIN_SKILLS_MATCH = 2  # At least 2 skills should match

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text

# Skill matching function
def check_skills(text):
    matches = [skill for skill in REQUIRED_SKILLS if skill in text]
    return len(matches)

# Streamlit app
st.title("Resume Screening Mini Project")
st.write("Paste the resume below to predict the category and check accept/reject status.")

resume_text = st.text_area("Enter Resume Text Here", height=300)

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text!")
    else:
        # Preprocess
        clean_text = preprocess_text(resume_text)
        # Vectorize
        vector_input = vectorizer.transform([clean_text])
        # Predict category
        category = model.predict(vector_input)[0]
        st.success(f"Predicted Category: **{category}**")

        matched_categories = ['Data Science','HR','Advocate', 'Arts', 'Web Designing', 'Mechanical Engineer', 
                          'Sales', 'Health and fitness', 'Civil Engineer',
                          'Java Developer', 'Business Analyst', 'SAP Developer', 'Automation Testing',
                          'Electrical Engineering', 'Operations Manager', 'Python Developer',
                          'DevOps Engineer', 'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
                          'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']


        # Suppose your model predicts a number
        predicted_number = model.predict(vector_input)[0]  # e.g., 0, 1, 2 ...
        
        # Map to name
        predicted_category_name = matched_categories[predicted_number]
        
        st.success(f"Predicted Category: **{predicted_category_name}**")

        st.subheader("Matched Category")
        st.write(f"- {predicted_category_name}")




        # Check criteria
        skills_matched = check_skills(clean_text)
        if skills_matched >= MIN_SKILLS_MATCH:
            st.success(f"Resume Status: **ACCEPTED** ({skills_matched} skills matched)")
        else:
            st.error(f"Resume Status: **REJECTED** ({skills_matched} skills matched)")

# Example: your model predicts which categories matched











