import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

import faiss
import numpy as np
import json
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

if "summary_document" not in st.session_state:
    st.session_state.summary_document = None

if "study_notes" not in st.session_state:
    st.session_state.study_notes = None

if "notes_document" not in st.session_state:
    st.session_state.notes_document = None

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_document" not in st.session_state:
    st.session_state.quiz_document = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# =========================================================
# HEADER
# =========================================================

st.title("📚 AI Study Assistant")

st.write(
    "Upload multiple PDFs and study using AI-powered summaries, "
    "notes, questions and quizzes."
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "📤 Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 AI Study Assistant")

    st.markdown("---")

    st.subheader("📂 Uploaded Documents")

    if uploaded_files:

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

        st.success(
            f"{len(uploaded_files)} PDF(s) loaded"
        )

    else:

        st.info(
            "No PDFs uploaded yet."
        )


    st.markdown("---")


    # =====================================================
    # DOCUMENT SELECTION
    # =====================================================

    st.subheader("📑 Select Document")

    if uploaded_files:

        document_options = [
            "All Documents"
        ]

        document_options.extend(
            [
                file.name
                for file in uploaded_files
            ]
        )

        selected_document = st.selectbox(
            "Choose a PDF:",
            document_options
        )

    else:

        selected_document = (
            "All Documents"
        )


    st.markdown("---")


    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN PDF PROCESSING
# =========================================================

if uploaded_files:


    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully! ✅"
    )


    # =====================================================
    # READ PDFS
    # =====================================================

    pages_data = []

    total_pages = 0


    for uploaded_file in uploaded_files:

        try:

            reader = PdfReader(
                uploaded_file
            )

            total_pages += len(
                reader.pages
            )


            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                page_text = (
                    page.extract_text()
                )


                if (
                    page_text
                    and page_text.strip()
                ):

                    pages_data.append(
                        {
                            "filename": (
                                uploaded_file.name
                            ),

                            "page": (
                                page_number
                            ),

                            "text": (
                                page_text
                            )
                        }
                    )


        except Exception:

            st.error(
                f"❌ Could not read "
                f"{uploaded_file.name}"
            )


    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    col1, col2 = st.columns(
        2
    )


    with col1:

        st.metric(
            "📚 Total PDFs",
            len(uploaded_files)
        )


    with col2:

        st.metric(
            "📄 Total Pages",
            total_pages
        )


    # =====================================================
    # CHECK PDF TEXT
    # =====================================================

    if not pages_data:

        st.error(
            "❌ No readable text was found "
            "in the uploaded PDFs."
        )

        st.info(
            "The PDF may contain scanned images "
            "instead of selectable text."
        )

        st.stop()


    # =====================================================
    # TEXT CHUNKING
    # =====================================================

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    )


    chunks = []

    chunk_sources = []


    for page_data in pages_data:

        page_chunks = (
            text_splitter.split_text(
                page_data["text"]
            )
        )


        for chunk in page_chunks:

            if chunk.strip():

                chunks.append(
                    chunk
                )

                chunk_sources.append(
                    {
                        "filename": (
                            page_data["filename"]
                        ),

                        "page": (
                            page_data["page"]
                        )
                    }
                )


    # =====================================================
    # DOCUMENT PROCESSING INFORMATION
    # =====================================================

    with st.expander(
        "✂️ Document Processing Information"
    ):

        st.write(
            f"Total text chunks: "
            f"**{len(chunks)}**"
        )

        st.write(
            f"Total readable pages: "
            f"**{len(pages_data)}**"
        )


    # =====================================================
    # LOAD EMBEDDING MODEL
    # =====================================================

    @st.cache_resource
    def load_embedding_model():

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    model = (
        load_embedding_model()
    )


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    with st.spinner(
        "🧠 Creating document embeddings..."
    ):

        embeddings = model.encode(
            chunks,
            show_progress_bar=False
        )


    embeddings = np.array(
        embeddings
    ).astype(
        "float32"
    )


    # =====================================================
    # CREATE FAISS DATABASE
    # =====================================================

    dimension = (
        embeddings.shape[1]
    )


    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(
        embeddings
    )


    # =====================================================
    # INITIALIZE OLLAMA
    # =====================================================

    llm = OllamaLLM(
        model="llama3.2"
    )


    # =====================================================
    # SELECTED DOCUMENT DATA
    # =====================================================

    selected_chunks = []

    selected_sources = []


    for chunk, source in zip(
        chunks,
        chunk_sources
    ):

        if (
            selected_document
            == "All Documents"

            or

            source["filename"]
            == selected_document
        ):

            selected_chunks.append(
                chunk
            )

            selected_sources.append(
                source
            )


    # =========================================================
    # AI PDF SUMMARY
    # =========================================================

    st.markdown(
        "---"
    )

    st.subheader(
        "📄 AI PDF Summary"
    )


    if (
        selected_document
        == "All Documents"
    ):

        st.info(
            "📚 Summary will be generated "
            "from all uploaded documents."
        )

    else:

        st.info(
            f"📄 Summary will be generated "
            f"from: **{selected_document}**"
        )


    if st.button(
        "✨ Generate Summary",
        use_container_width=True
    ):

        if not selected_chunks:

            st.warning(
                "No readable content found "
                "in the selected document."
            )

        else:

            max_summary_chunks = min(
                12,
                len(selected_chunks)
            )


            summary_chunks = (
                selected_chunks[
                    :max_summary_chunks
                ]
            )


            summary_context = (
                "\n\n".join(
                    summary_chunks
                )
            )


            summary_prompt = f"""
You are an AI Study Assistant.

Create a clear and useful summary using ONLY the
information provided in the PDF content below.

Do not add outside knowledge.
Do not invent information.

Your response should contain these sections:

# 📌 OVERVIEW

Give a short overview of the document.

# 📚 IMPORTANT POINTS

Give the most important points as bullet points.

# 🔑 KEY CONCEPTS

List important concepts or topics mentioned.

# 📝 QUICK REVISION

Give short revision points useful for students.

Keep the answer easy to understand.

DOCUMENT:
{selected_document}

PDF CONTENT:
{summary_context}
"""


            with st.spinner(
                "🤖 AI is generating the summary..."
            ):

                try:

                    summary = (
                        llm.invoke(
                            summary_prompt
                        )
                    )


                    st.session_state.summary_result = (
                        summary
                    )


                    st.session_state.summary_document = (
                        selected_document
                    )


                except Exception:

                    st.session_state.summary_result = None


                    st.error(
                        "❌ Could not connect to Ollama."
                    )


                    st.info(
                        "Make sure Ollama is running "
                        "and llama3.2 is installed."
                    )


    # =========================================================
    # DISPLAY SUMMARY
    # =========================================================

    if (
        st.session_state.summary_result

        and

        st.session_state.summary_document
        == selected_document
    ):

        st.success(
            "Summary generated successfully! ✅"
        )


        st.markdown(
            st.session_state.summary_result
        )


    # =========================================================
    # GENERATE STUDY NOTES
    # =========================================================

    st.markdown(
        "---"
    )

    st.subheader(
        "📝 Generate Study Notes"
    )


    if (
        selected_document
        == "All Documents"
    ):

        st.info(
            "📚 Study notes will be generated "
            "from all uploaded documents."
        )

    else:

        st.info(
            f"📄 Study notes will be generated "
            f"from: **{selected_document}**"
        )


    if st.button(
        "📝 Generate Study Notes",
        use_container_width=True
    ):

        if not selected_chunks:

            st.warning(
                "No readable content found "
                "in the selected document."
            )

        else:

            max_notes_chunks = min(
                15,
                len(selected_chunks)
            )


            notes_chunks = (
                selected_chunks[
                    :max_notes_chunks
                ]
            )


            notes_context = (
                "\n\n".join(
                    notes_chunks
                )
            )


            notes_prompt = f"""
You are an AI Study Assistant.

Create clear, organized and student-friendly study notes
using ONLY the information provided in the PDF content.

IMPORTANT RULES:

1. Do not add outside knowledge.
2. Do not invent information.
3. Use only the provided PDF content.
4. Keep explanations simple.
5. Use headings and bullet points.
6. Make the notes useful for exam revision.

Your answer MUST contain these sections:

# 📌 IMPORTANT POINTS

Write the most important points as bullet points.

# 🔑 KEY CONCEPTS

Explain the main concepts in simple language.

# 📝 QUICK REVISION NOTES

Create short revision notes that are easy to remember.

# ⭐ IMPORTANT KEYWORDS

List important keywords and terms.

# 🎯 EXAM FOCUS

List important topics for exam preparation.

DOCUMENT:
{selected_document}

PDF CONTENT:
{notes_context}
"""


            with st.spinner(
                "🧠 AI is creating your study notes..."
            ):

                try:

                    notes = (
                        llm.invoke(
                            notes_prompt
                        )
                    )


                    st.session_state.study_notes = (
                        notes
                    )


                    st.session_state.notes_document = (
                        selected_document
                    )


                except Exception:

                    st.session_state.study_notes = None


                    st.error(
                        "❌ Could not connect to Ollama."
                    )


                    st.info(
                        "Make sure Ollama is running "
                        "and llama3.2 is installed."
                    )


    # =========================================================
    # DISPLAY STUDY NOTES
    # =========================================================

    if (
        st.session_state.study_notes

        and

        st.session_state.notes_document
        == selected_document
    ):

        st.success(
            "Study notes generated successfully! ✅"
        )


        st.markdown(
            st.session_state.study_notes
        )


    # =========================================================
    # MCQ QUIZ GENERATOR
    # =========================================================

    st.markdown(
        "---"
    )

    st.subheader(
        "❓ AI MCQ Quiz"
    )


    if (
        selected_document
        == "All Documents"
    ):

        st.info(
            "📚 Quiz will be generated "
            "from all uploaded documents."
        )

    else:

        st.info(
            f"📄 Quiz will be generated "
            f"from: **{selected_document}**"
        )


    # =====================================================
    # GENERATE QUIZ
    # =====================================================

    if st.button(
        "🎯 Generate MCQ Quiz",
        use_container_width=True
    ):

        if not selected_chunks:

            st.warning(
                "No readable content found "
                "in the selected document."
            )

        else:

            with st.spinner(
                "🧠 AI is creating your quiz..."
            ):

                try:

                    max_quiz_chunks = min(
                        12,
                        len(selected_chunks)
                    )


                    quiz_chunks = (
                        selected_chunks[
                            :max_quiz_chunks
                        ]
                    )


                    quiz_context = (
                        "\n\n".join(
                            quiz_chunks
                        )
                    )


                    quiz_prompt = f"""
You are an AI Study Assistant.

Create exactly 5 multiple choice questions using ONLY
the PDF content provided below.

Do not use outside knowledge.
Do not invent information.

Return ONLY valid JSON.

Do not use markdown.

Do not add any explanation before or after the JSON.

Use exactly this format:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A"
  }}
]

IMPORTANT RULES:

1. Create exactly 5 questions.
2. Every question must have exactly 4 options.
3. The answer must exactly match one option.
4. Questions must be based only on the PDF.
5. Make questions useful for students.

DOCUMENT:
{selected_document}

PDF CONTENT:
{quiz_context}
"""


                    quiz_response = (
                        llm.invoke(
                            quiz_prompt
                        )
                    )


                    # Clean markdown if AI adds it
                    cleaned_response = (
                        quiz_response.strip()
                    )


                    cleaned_response = re.sub(
                        r"```json",
                        "",
                        cleaned_response
                    )


                    cleaned_response = re.sub(
                        r"```",
                        "",
                        cleaned_response
                    )


                    cleaned_response = (
                        cleaned_response.strip()
                    )


                    # Extract JSON array if extra text exists
                    json_match = re.search(
                        r"\[.*\]",
                        cleaned_response,
                        re.DOTALL
                    )


                    if json_match:

                        cleaned_response = (
                            json_match.group()
                        )


                    quiz_data = json.loads(
                        cleaned_response
                    )


                    # Validate quiz
                    valid_questions = []


                    for mcq in quiz_data:

                        if (
                            isinstance(
                                mcq,
                                dict
                            )

                            and

                            "question" in mcq

                            and

                            "options" in mcq

                            and

                            "answer" in mcq

                            and

                            len(
                                mcq["options"]
                            ) == 4

                            and

                            mcq["answer"]
                            in mcq["options"]
                        ):

                            valid_questions.append(
                                mcq
                            )


                    if valid_questions:

                        st.session_state.quiz_questions = (
                            valid_questions
                        )


                        st.session_state.quiz_document = (
                            selected_document
                        )


                        st.session_state.quiz_submitted = (
                            False
                        )


                        # Remove old answers
                        for key in list(
                            st.session_state.keys()
                        ):

                            if key.startswith(
                                "quiz_answer_"
                            ):

                                del st.session_state[
                                    key
                                ]


                        st.rerun()


                    else:

                        st.error(
                            "❌ AI could not generate "
                            "valid quiz questions. "
                            "Please generate again."
                        )


                except Exception as e:

                    st.error(
                        "❌ Could not generate "
                        "the quiz. Please try again."
                    )


                    st.info(
                        "Make sure Ollama is running."
                    )


    # =====================================================
    # DISPLAY QUIZ
    # =====================================================

    if (
        st.session_state.quiz_questions

        and

        st.session_state.quiz_document
        == selected_document
    ):

        st.success(
            "MCQ Quiz generated successfully! 🎉"
        )


        with st.form(
            "mcq_quiz_form"
        ):


            for question_number, mcq in enumerate(
                st.session_state.quiz_questions,
                start=1
            ):


                st.markdown(
                    f"### Question {question_number}"
                )


                st.write(
                    mcq["question"]
                )


                st.radio(
                    "Choose your answer:",
                    mcq["options"],
                    key=(
                        f"quiz_answer_"
                        f"{question_number}"
                    ),
                    index=None
                )


                st.divider()


            submitted = (
                st.form_submit_button(
                    "📊 Submit Quiz",
                    use_container_width=True
                )
            )


        # =================================================
        # CALCULATE RESULT
        # =================================================

        if submitted:

            st.session_state.quiz_submitted = (
                True
            )


        if (
            st.session_state.quiz_submitted
        ):


            score = 0


            total_questions = len(
                st.session_state.quiz_questions
            )


            st.markdown(
                "---"
            )


            st.subheader(
                "📊 Quiz Result"
            )


            for question_number, mcq in enumerate(
                st.session_state.quiz_questions,
                start=1
            ):


                user_answer = (
                    st.session_state.get(
                        f"quiz_answer_"
                        f"{question_number}"
                    )
                )


                correct_answer = (
                    mcq["answer"]
                )


                if (
                    user_answer
                    == correct_answer
                ):

                    score += 1


                    st.success(
                        f"Question "
                        f"{question_number}: "
                        f"Correct! ✅"
                    )


                else:

                    st.error(
                        f"Question "
                        f"{question_number}: "
                        f"Incorrect ❌"
                    )


                    st.info(
                        f"Correct Answer: "
                        f"{correct_answer}"
                    )


            st.markdown(
                f"## 🎯 Your Score: "
                f"{score} / {total_questions}"
            )


            percentage = (
                score
                / total_questions
            ) * 100


            st.progress(
                int(percentage)
            )


            st.write(
                f"Percentage: "
                f"**{percentage:.0f}%**"
            )


            if percentage == 100:

                st.balloons()


                st.success(
                    "🏆 Excellent! "
                    "Perfect Score!"
                )


            elif percentage >= 70:

                st.success(
                    "👏 Great job! "
                    "Keep practicing!"
                )


            elif percentage >= 40:

                st.warning(
                    "👍 Good effort! "
                    "Revise and try again."
                )


            else:

                st.warning(
                    "📚 Keep studying "
                    "and try again!"
                )


    # =====================================================
    # GENERATE NEW QUIZ
    # =====================================================

    if (
        st.session_state.quiz_questions

        and

        st.session_state.quiz_document
        == selected_document
    ):

        if st.button(
            "🔄 Clear Quiz / Generate New Quiz",
            use_container_width=True
        ):


            st.session_state.quiz_questions = []


            st.session_state.quiz_document = None


            st.session_state.quiz_submitted = (
                False
            )


            for key in list(
                st.session_state.keys()
            ):

                if key.startswith(
                    "quiz_answer_"
                ):

                    del st.session_state[
                        key
                    ]


            st.rerun()


    # =========================================================
    # ASK QUESTIONS
    # =========================================================

    st.markdown(
        "---"
    )

    st.subheader(
        "💬 Ask Questions"
    )


    if (
        selected_document
        == "All Documents"
    ):

        st.info(
            "📚 Searching in: "
            "**All Documents**"
        )

    else:

        st.info(
            f"📄 Searching in: "
            f"**{selected_document}**"
        )


    question = st.chat_input(
        "Ask something about your selected PDF..."
    )


    # =====================================================
    # PROCESS QUESTION
    # =====================================================

    if question:


        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        if not selected_chunks:

            answer = (
                "I could not find this information "
                "in the selected PDF."
            )


            source_pages = []


            with st.chat_message(
                "assistant"
            ):

                st.warning(
                    answer
                )


        else:


            # =================================================
            # CREATE SELECTED DOCUMENT EMBEDDINGS
            # =================================================

            selected_embeddings = (
                model.encode(
                    selected_chunks,
                    show_progress_bar=False
                )
            )


            selected_embeddings = np.array(
                selected_embeddings
            ).astype(
                "float32"
            )


            selected_index = (
                faiss.IndexFlatL2(
                    dimension
                )
            )


            selected_index.add(
                selected_embeddings
            )


            # =================================================
            # QUESTION EMBEDDING
            # =================================================

            question_embedding = (
                model.encode(
                    [question]
                )
            )


            question_embedding = np.array(
                question_embedding
            ).astype(
                "float32"
            )


            # =================================================
            # FAISS SEARCH
            # =================================================

            k = min(
                5,
                selected_index.ntotal
            )


            distances, indices = (
                selected_index.search(
                    question_embedding,
                    k
                )
            )


            relevant_chunks = []

            source_pages = []


            for distance, i in zip(
                distances[0],
                indices[0]
            ):

                if i >= 0:

                    relevant_chunks.append(
                        selected_chunks[i]
                    )


                    source_pages.append(
                        selected_sources[i]
                    )


            # =================================================
            # BUILD CONTEXT
            # =================================================

            context_parts = []


            for chunk, source in zip(
                relevant_chunks,
                source_pages
            ):


                context_parts.append(
                    f"""
Source: {source['filename']}
Page: {source['page']}

Content:
{chunk}
"""
                )


            context = (
                "\n\n".join(
                    context_parts
                )
            )


            # =================================================
            # AI PROMPT
            # =================================================

            prompt = f"""
You are an AI Study Assistant.

Answer the user's question using ONLY the
information provided in the selected PDF context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not make up information.
3. Answer only from the PDF context.
4. If the answer is not present, say exactly:

"I could not find this information in the selected PDF."

5. Give a clear and concise answer.

SELECTED DOCUMENT:
{selected_document}

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


            # =================================================
            # GENERATE ANSWER
            # =================================================

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "🤖 AI is thinking..."
                ):

                    try:

                        answer = (
                            llm.invoke(
                                prompt
                            )
                        )


                    except Exception:

                        answer = (
                            "Could not connect to Ollama. "
                            "Please make sure Ollama is running."
                        )


                        st.error(
                            answer
                        )


                if answer:

                    st.write(
                        answer
                    )


                    # =============================================
                    # SHOW SOURCES
                    # =============================================

                    if source_pages:


                        st.markdown(
                            "**📚 Sources:**"
                        )


                        shown_sources = set()


                        for source in source_pages:


                            source_key = (
                                source["filename"],
                                source["page"]
                            )


                            if (
                                source_key
                                not in shown_sources
                            ):


                                st.caption(
                                    f"📄 "
                                    f"{source['filename']} "
                                    f"| Page: "
                                    f"{source['page']}"
                                )


                                shown_sources.add(
                                    source_key
                                )


        # =====================================================
        # SAVE CHAT HISTORY
        # =====================================================

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": source_pages
            }
        )


    # =========================================================
    # PREVIOUS QUESTIONS
    # =========================================================

    if (
        st.session_state.chat_history
    ):

        st.markdown(
            "---"
        )


        st.subheader(
            "🕘 Previous Questions"
        )


        for chat in (
            st.session_state.chat_history
        ):


            with st.container():


                st.markdown(
                    f"**👤 You:** "
                    f"{chat['question']}"
                )


                st.markdown(
                    f"**🤖 AI:** "
                    f"{chat['answer']}"
                )


                if (
                    chat["sources"]
                ):


                    st.markdown(
                        "**📚 Sources:**"
                    )


                    shown_sources = set()


                    for source in (
                        chat["sources"]
                    ):


                        source_key = (
                            source["filename"],
                            source["page"]
                        )


                        if (
                            source_key
                            not in shown_sources
                        ):


                            st.caption(
                                f"📄 "
                                f"{source['filename']} "
                                f"| Page: "
                                f"{source['page']}"
                            )


                            shown_sources.add(
                                source_key
                            )


                st.divider()


# =========================================================
# NO PDF UPLOADED
# =========================================================

else:

    st.info(
        "👆 Please upload one or more PDF files to start."
    )
# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <h4>💻 Developed by Swarup Dhere</h4>
        <p>📚 AI Study Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)
