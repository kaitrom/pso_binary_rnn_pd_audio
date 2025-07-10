import subprocess
import pypandoc

# Faz o download automático do Pandoc embutido
pypandoc.download_pandoc()

notebook = "relatorio_final.ipynb"
markdown_file = "relatorio_final.md"
docx_file = "relatorio_final.docx"

# Converter notebook para markdown
subprocess.run([
    "jupyter", "nbconvert", "--to", "markdown", notebook
], check=True)

# Converter markdown para docx
output = pypandoc.convert_file(markdown_file, "docx", outputfile=docx_file)
print(f"DOCX gerado com sucesso: {docx_file}")
