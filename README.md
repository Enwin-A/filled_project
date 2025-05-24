# AI Assignment

Hi there 👋,

This is the take home assignment for Filed's AI engineer position. 
We recommend spending no more than 5–6 hours on it — we're not looking for perfection, but rather how you think and approach problems.

You can clone this repository into your github account and then complete it.

There is no set time to complete the assignment, but faster you complete higher the chances that the position is not filled by someone else. 

Once you're done, just reply back to the email you received with the link to your completed github repo and we'll get back to you shortly after.

PS: If its a private repo - please add atul@filed.com as the outside collaborator


## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Setup


0. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone this repo

2. Install dependencies using uv:
```bash
uv sync
```

3. Run:
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. You can now open the docs at http://0.0.0.0:8000/docs and run the endpoints

## API Endpoints

### Docs

http://0.0.0.0:8000/docs


### Document Classification
- `POST /classify` - Submit a document to be classified


## Task 


Your task is to complete the /classify endpoint
The endpoint should 

1. Take in a PDF file as an input - Use the sample documents provided under sample directory
2. Classify the PDF as one of 

- "1040"
- "W2"
- "1099
- "ID Card"
- "Handwritten note"
- "OTHER"

3. Also parse the year the document was issued
