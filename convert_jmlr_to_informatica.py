import re
import os

jmlr_path = r"C:\ingester_ops\argus\jmlr-style-file-master\jmlr-style-file-master\ARGUS_JMLR.tex"
info_path = r"C:\ingester_ops\argus\InformaticaPaper\author.tex"

def convert_latex():
    try:
        with open(jmlr_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract body
        body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
        body = body_match.group(1) if body_match else ""

        # Extract title
        title_m = re.search(r"\\title\{(.*?)\}", body, re.DOTALL)
        title_text = title_m.group(1) if title_m else "ARGUS: A Debate-Native Multi-Agent Architecture for Evidence-Based Reasoning"
        title_text = " ".join(title_text.split())

        # Extract abstract
        abstract_m = re.search(r"\\begin\{abstract\}%?\n*(.*?)\n*\\end\{abstract\}", body, re.DOTALL)
        abstract_text = abstract_m.group(1).strip() if abstract_m else "Abstract."

        # Clean up body
        body = re.sub(r"\\title\{.*?\}", "", body, flags=re.DOTALL)
        body = re.sub(r"\\author\{.*?\}", "", body, flags=re.DOTALL)
        body = re.sub(r"\\editor\{.*?\}", "", body, flags=re.DOTALL)
        body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.DOTALL)
        body = re.sub(r"\\maketitle", "", body, flags=re.DOTALL)

        # Extract preamble
        preamble_m = re.search(r"\\documentclass.*?\]\{article\}(.*?)\\begin\{document\}", text, re.DOTALL)
        preamble = preamble_m.group(1) if preamble_m else ""

        # Remove JMLR specifics from preamble
        preamble = re.sub(r"\\usepackage\[.*?\]\{jmlr2e\}", "", preamble, flags=re.DOTALL)
        preamble = re.sub(r"\\jmlrheading\{.*?\}", "", preamble, flags=re.DOTALL)
        preamble = re.sub(r"\\ShortHeadings\{.*?\}", "", preamble, flags=re.DOTALL)
        preamble = re.sub(r"\\firstpageno\{.*?\}", "", preamble, flags=re.DOTALL)

        # Build Informatica format string
        out_content = f"""\\documentclass[11pt,twoside]{{article}}
\\usepackage{{informat}}
\\usepackage{{epsfig}}

{preamble.strip()}

\\begin{{document}}
\\title{{{title_text}}}
\\author{{Nilesh Marathe, Kriti Srivastava, Namita Pulgam, \\\\\\\\ Ankush Pandey, Rishi Ghodawat and Ronit Mehta \\\\\\\\
SVKM's Dwarkadas J. Sanghvi College of Engineering \\\\\\\\
Mumbai, India \\\\\\\\
mehtaronit702@gmail.com}}
\\titleodd{{ARGUS: Debate-Native Multi-Agent Architecture}}
\\authoreven{{N. Marathe et al.}}
\\keywords{{Multi-Agent, Debate, Evidence-Based Reasoning, LLM}}
\\received{{April 1, 2026}}

\\abstract{{{abstract_text}}}
\\abstractSi{{}}

\\maketitle

{body.strip()}

\\end{{document}}
"""

        with open(info_path, "w", encoding="utf-8") as f:
            f.write(out_content)

        print(f"✅ Conversion complete!")
        print(f"Source mapped from: {jmlr_path}")
        print(f"Target file generated at: {info_path}")

    except Exception as e:
        print(f"Error during mapping: {e}")

if __name__ == "__main__":
    convert_latex()