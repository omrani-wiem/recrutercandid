from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import os

output_path = "samples/example_cv.pdf"
os.makedirs("samples", exist_ok=True)

c = canvas.Canvas(output_path, pagesize=letter)
c.setFont("Helvetica", 13)
text = c.beginText(50, 700)
text.textLine("Jean Dupont")
text.textLine("Data Engineer Senior")
text.textLine("")
text.textLine("Expérience professionnelle :")
text.textLine("- 6 ans en data engineering, pipelines ETL, Azure, Airflow")
text.textLine("- Conception de solutions de données et automatisation de flux")
text.textLine("")
text.textLine("Compétences : Python, SQL, Airflow, Azure, AWS, Spark")
text.textLine("Formation : Master en informatique")
text.textLine("")
text.textLine("Langues : français, anglais")

c.drawText(text)
c.showPage()
c.save()
print(f"PDF généré : {output_path}")
