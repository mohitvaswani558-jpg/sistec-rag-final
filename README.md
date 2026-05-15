# Study Buddy: Document-Based Q&A Assistant

Study Buddy is a strict document-based Q&A assistant designed for students. It answers questions based solely on the content of an uploaded PDF document, rejecting any queries that cannot be answered from the document.

## Features

- Extracts text from PDF documents
- Uses retrieval-augmented generation (RAG) to find relevant information
- Provides concise answers with source citations
- Strictly adheres to document content only

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   - Create a `.env` file in the project directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

Run the script with the PDF file path and your question:

```
python main.py path/to/your/document.pdf "Your question here"
```

Example:
```
python main.py notes.pdf "What is the capital of France?"
```

If the question can be answered from the document, you'll get:
```
✅ Answer: [Concise answer]
📄 Source:
- Chunk 1: "[relevant passage]"
- Chunk 2: "[another passage]"
```

If the question is outside the scope:
```
❌ This question is outside the scope of the uploaded document. I can only answer questions based on its content.
```

## Requirements

- Python 3.7+
- OpenAI API key
- PDF document with extractable text

## Notes

- The assistant only uses information from the provided PDF document
- No external knowledge is incorporated
- Answers are kept concise and direct
- Source chunks are displayed for verification