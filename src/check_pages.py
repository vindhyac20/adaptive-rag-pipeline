from pypdf import PdfReader

reader = PdfReader("data/MachineLearningTomMitchell.pdf")
for i in range(10, 20):
    text = reader.pages[i].extract_text() or ""
    marker = " <-- CUT (before page 15)" if i < 15 else ""
    print(f"--- PDF page {i}{marker} ---")
    print(text[:200].replace("\n", " "))
    print()