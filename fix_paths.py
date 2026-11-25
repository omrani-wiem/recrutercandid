from pathlib import Path

p = Path("data/candidates.json")
text = p.read_text(encoding="utf-8")
text = text.replace("services\\\\data", "data")
p.write_text(text, encoding="utf-8")
print("chemins corrigés")