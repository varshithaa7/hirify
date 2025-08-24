import streamlit as st
import ollama
import json
import docx
import fitz  # PyMuPDF for PDF reading

# ---------------------- Helper Functions ----------------------
def call_ollama(prompt, system_message="You are a helpful assistant"):
    response = ollama.chat(model="llama3.2:1b", messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ])
    return response["message"]["content"]

def read_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    else:
        return "Unsupported file type"

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="JD Assistant", layout="wide")

st.sidebar.title("JD Assistant Modules")
module = st.sidebar.radio("Select Module", [
    "JD Generation",
    "JD Extraction",
    "JD Reasoning",
    "JD vs Resume Matching"
])

# ---------------------- JD Generation ----------------------
if module == "JD Generation":
    st.header("📄 Job Description Generator")
    user_prompt = st.text_area("Enter prompt for JD generation:")
    if st.button("Generate JD"):
        jd_json = call_ollama(
            f"Generate a Job Description in strict JSON format based on: {user_prompt}. Include role, skills, experience, responsibilities."
        )
        try:
            parsed = json.loads(jd_json)
            st.json(parsed)
        except:
            st.write(jd_json)

# ---------------------- JD Extraction ----------------------
elif module == "JD Extraction":
    st.header("🔍 Job Description Extraction")
    jd_file = st.file_uploader("Upload Job Description (txt/pdf/docx)")

    # Initialize and manage session history per uploaded file
    if "jd_extraction_history" not in st.session_state:
        st.session_state.jd_extraction_history = []
    if "jd_extraction_current_file" not in st.session_state:
        st.session_state.jd_extraction_current_file = None

    if jd_file is not None:
        if st.session_state.jd_extraction_current_file != jd_file.name:
            st.session_state.jd_extraction_current_file = jd_file.name
            st.session_state.jd_extraction_history = []

        with st.form("jd_extraction_form", clear_on_submit=True):
            query = st.text_input(
                "Ask factual question (e.g., skills required, experience level):",
                placeholder="Type your question and press Enter",
            )
            submitted = st.form_submit_button("Enter")

        if submitted and query:
            jd_text = read_file(jd_file)
            response = call_ollama(
                f"Job Description: {jd_text}\n\nAnswer only factual extraction questions. Question: {query}"
            )
            st.session_state.jd_extraction_history.append({
                "question": query,
                "answer": response,
            })

    # Display history of prompts and answers
    if st.session_state.jd_extraction_history:
        st.subheader("History")
        for idx, qa in enumerate(st.session_state.jd_extraction_history, start=1):
            st.markdown(f"**Q{idx}:** {qa['question']}")
            st.markdown(f"**A{idx}:** {qa['answer']}")

# ---------------------- JD Reasoning ----------------------
elif module == "JD Reasoning":
    st.header("🤔 JD Reasoning")
    jd_file = st.file_uploader("Upload Job Description (txt/pdf/docx)")

    # Initialize and manage session history per uploaded file
    if "jd_reasoning_history" not in st.session_state:
        st.session_state.jd_reasoning_history = []
    if "jd_reasoning_current_file" not in st.session_state:
        st.session_state.jd_reasoning_current_file = None

    if jd_file is not None:
        if st.session_state.jd_reasoning_current_file != jd_file.name:
            st.session_state.jd_reasoning_current_file = jd_file.name
            st.session_state.jd_reasoning_history = []

        with st.form("jd_reasoning_form", clear_on_submit=True):
            query = st.text_input(
                "Ask reasoning-based question (e.g., Why is X skill required?):",
                placeholder="Type your question and press Enter",
            )
            submitted = st.form_submit_button("Enter")

        if submitted and query:
            jd_text = read_file(jd_file)
            response = call_ollama(
                f"Job Description: {jd_text}\n\nAnswer reasoning question: {query}"
            )
            st.session_state.jd_reasoning_history.append({
                "question": query,
                "answer": response,
            })

    # Display history of prompts and answers
    if st.session_state.get("jd_reasoning_history"):
        st.subheader("History")
        for idx, qa in enumerate(st.session_state.jd_reasoning_history, start=1):
            st.markdown(f"**Q{idx}:** {qa['question']}")
            st.markdown(f"**A{idx}:** {qa['answer']}")

# ---------------------- JD vs Resume Matching ----------------------
elif module == "JD vs Resume Matching":
    st.header("📑 JD vs Resume Matching")
    jd_file = st.file_uploader("Upload Job Description (txt/pdf/docx)", key="jd")
    resumes = st.file_uploader("Upload Resume(s) (txt/pdf/docx)", accept_multiple_files=True, key="resumes")

    # Initialize and manage session history per uploaded JD + resumes set
    if "jd_match_history" not in st.session_state:
        st.session_state.jd_match_history = []
    if "jd_match_signature" not in st.session_state:
        st.session_state.jd_match_signature = None

    if jd_file is not None and resumes:
        resumes_names = sorted([r.name for r in resumes])
        signature = f"{jd_file.name}|{','.join(resumes_names)}"

        if st.session_state.jd_match_signature != signature:
            st.session_state.jd_match_signature = signature
            st.session_state.jd_match_history = []

        with st.form("jd_match_form", clear_on_submit=True):
            query = st.text_input(
                "Ask query about candidate matching:",
                placeholder="Type your question and press Enter",
            )
            submitted = st.form_submit_button("Enter")

        if submitted and query:
            jd_text = read_file(jd_file)
            resumes_text = [read_file(r) for r in resumes]
            combined_resume_text = "\n\n".join([f"Resume {i+1}:\n{txt}" for i, txt in enumerate(resumes_text)])

            prompt = f"""
            Job Description: {jd_text}

            Resumes:
            {combined_resume_text}

            Task: Compare resumes against the JD. Identify best matching candidate(s). 
            Provide reasons for selection. If none match, clearly state 'No suitable candidate found'.
            User Query: {query}
            """

            response = call_ollama(prompt)
            st.session_state.jd_match_history.append({
                "question": query,
                "answer": response,
            })

    # Display history of prompts and answers
    if st.session_state.get("jd_match_history"):
        st.subheader("History")
        for idx, qa in enumerate(st.session_state.jd_match_history, start=1):
            st.markdown(f"**Q{idx}:** {qa['question']}")
            st.markdown(f"**A{idx}:** {qa['answer']}")
