import fitz

src = fitz.open("MedCalc (1).pdf")
out = fitz.open()
for page in src:
    pix = page.get_pixmap(dpi=150)
    new = out.new_page(width=page.rect.width, height=page.rect.height)
    new.insert_image(new.rect, pixmap=pix)
out.save("cv_scanne.pdf")
print("cv_scanne.pdf créé")