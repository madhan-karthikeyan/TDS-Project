# TDS Virtual TA — FastAPI-based RAG with OCR Assistant

This project is a backend API application designed as a virtual teaching assistant for the IIT Madras Tools in Data Science (TDS) course. It uses FastAPI along with LangChain and vector-based retrieval to answer student questions using course content and forum discussions. It also supports OCR-based question extraction from uploaded images.

## Features

- Retrieval-Augmented Generation (RAG) using LangChain and Chroma vector store
- Custom prompt logic with strict content constraints and fallback behaviors
- RESTful API with image-based OCR support using pytesseract
- Automatic extraction of source URLs and inline link filtering
- Configurable with API tokens for external model providers

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/tds-virtual-ta.git
cd tds-virtual-ta
```

### Set Up Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export AIPIPE_BEARER_TOKEN=your_token_here
```

### Run the Application

```bash
uvicorn server:app --reload --port 8000
```

## API Usage

### Endpoint

```
POST /api/
```

### JSON Request Body

```json
{
  "question": "What is the scoring if I get 10/10 and a bonus?",
  "image": "data:image/jpeg;base64,... (optional)"
}
```

### JSON Response Structure

```json
{
  "answer": "110",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/...",
      "text": "Clarification on bonus scoring"
    }
  ],
  "debug_info": {
    "total_retrieved": 20,
    "question_used": "...",
    "raw_answer_was_json": false
  }
}
```

## Testing with Promptfoo

This project supports prompt-level evaluations using Promptfoo.

```bash
npx promptfoo test
```

## Docker Support (Optional)

To run the application in a Docker container with OCR enabled:

```bash
docker build -t tds-ta .
docker run -p 8000:8000 --env AIPIPE_BEARER_TOKEN=your_token_here tds-ta
```

## License

This project is licensed under the MIT License.