 # services/strategy_gen.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_strategy_pdf(text, output="media/strategy.pdf"):
    c = canvas.Canvas(output, pagesize=letter)
    text_object = c.beginText(40, 750)
    for line in text.split('\n'):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
