# SuccessGuideAI

Overview

SuccessGuideAI is an AI-powered assistant that helps students better understand their course materials and explore personalized career guidance. Users can upload course-related PDF or TXT files and ask contextual questions. The app uses GPT-4, LangChain, HuggingFace Embeddings, and ChromaDB for intelligent, personalized responses.

Features

•	Upload course materials in PDF or TXT format
•	Ask questions like: 'How can I succeed in this course?' or 'What careers match my interests in AI and design?'
•	Personalized guidance based on your interests
•	GPT-4 conversational chat using LangChain
•	Vector search powered by HuggingFace and ChromaDB
•	Clean web interface built with Streamlit

Technologies Used

•	Python 3
•	Streamlit
•	OpenAI GPT-4 via LangChain
•	ChromaDB (persistent vector database)
•	HuggingFace Embeddings (multi-qa-MiniLM-L6-cos-v1)
•	PyPDF2
•	chardet (for file encoding detection)

Setup

•	Clone the repository:
•	Install the required packages:
  pip install -r requirements.txt
•	Add your OpenAI API key in app.py:
  openai_api_key = "your-api-key-here"
•	Run the application:
  streamlit run app.py

Example Use Cases

•	Upload your 'Introduction to AI' syllabus and ask:
  - 'Which topics should I focus on?'
  - 'Are there related job roles for my interest in education?'
•	Upload your programming objectives and ask:
  - 'How can I practice these skills?'
  - 'What careers align with coding and creativity?'

Future Improvements

•	Support for DOCX files
•	Multi-user session support
•	Dashboard to visualize skills and learning goals
•	Save and load previous conversations

Created by Tuğçe Ünlü
