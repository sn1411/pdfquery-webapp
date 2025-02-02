# pdfquery-webapp

# Introduction
The AskPDF App 📚 is a Python-based tool that enables users to interact with multiple PDF documents through natural language. You can ask questions about the contents of the PDFs, and the app will generate relevant answers based on the information in the documents. The app uses a language model to ensure accurate responses, but it will only provide answers related to the loaded PDFs.

# How It Works
  - The application follows these steps to provide responses to your questions:
  - PDF Loading: The app reads multiple PDF documents and extracts their text content.
  - Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.
  - Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
  - Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
  - Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.
