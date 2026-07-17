from pypdf import PdfReader

reader = PdfReader("uploads/notes.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(text[:3000])