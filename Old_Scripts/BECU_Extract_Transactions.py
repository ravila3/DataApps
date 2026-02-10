import fitz  # PyMuPDF
import pandas as pd
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

STATIC_DIRECTORY = 'C:/static/'

# Ensure the directory exists
if not os.path.exists(STATIC_DIRECTORY):
    os.makedirs(STATIC_DIRECTORY)

# Mount the static directory
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Endpoint to serve a specific static file
@app.get("/static-file/{file_path:path}")
async def serve_static_file(file_path: str):
    full_path = os.path.join(STATIC_DIRECTORY, file_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def extract_transactions(pdf_path):
    transactions = []
    document = fitz.open(pdf_path)

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text = page.get_text("text")

        lines = text.split("\n")
        for line in lines:
            if is_transaction_line(line):
                transactions.append(parse_transaction(line))

    df = pd.DataFrame(transactions, columns=["Date", "Description", "Amount"])
    return df

def is_transaction_line(line):
    # Logic to identify transaction lines, adjust as needed
    return any(char.isdigit() for char in line.split()[:3])

def parse_transaction(line):
    # Logic to parse a transaction line into date, description, and amount
    parts = line.split()
    date = parts[0]
    amount = parts[-1]
    description = " ".join(parts[1:-1])
    return [date, description, amount]

# Define the PDF path without extra quotes
pdf_path = "C:\\Users\\gadab\\OneDrive\\Documents\\Separation\\BECU Statements\\BECU-Statement-2005-12-09.PDF"
transactions_df = extract_transactions(pdf_path)
print(transactions_df)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

