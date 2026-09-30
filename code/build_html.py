"""
build_html.py
=============
Convert whitepaper.md into a single self-contained whitepaper.html:
  * LaTeX math ($...$ / $$...$$) rendered by MathJax (CDN),
  * figures embedded as base64 (portable single file),
  * GitHub-style pipe tables, code fences, blockquotes, lists.

This is a focused converter for the subset of Markdown used in the paper;
it is not a general Markdown engine.  Run:  python3 build_html.py
"""
from __future__ import annotations
import base64, html, os, re, mimetypes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MD = os.path.join(ROOT, "whitepaper.md")
OUT = os.path.join(ROOT, "whitepaper.html")

math_store: list[str] = []


def protect_math(text: str) -> str:
    """Replace $$...$$ and $...$ with placeholders so inline formatting and
    HTML-escaping never touch the TeX."""
    def repl_block(m):
        math_store.append(("block", m.group(1)))
        return f"\x00MATH{len(math_store)-1}\x00"

    def repl_inline(m):
        math_store.append(("inline", m.group(1)))
        return f"\x00MATH{len(math_store)-1}\x00"

    text = re.sub(r"(?<!\\)\$\$(.+?)\$\$", repl_block, text, flags=re.S)
    # inline $...$ but never when the $ is backslash-escaped (\$ = literal dollar)
    text = re.sub(r"(?<![\$\\])\$(?!\$)(.+?)(?<![\$\\])\$(?!\$)", repl_inline, text)
    return text


def restore_math(text: str) -> str:
    def repl(m):
        kind, tex = math_store[int(m.group(1))]
        tex = html.escape(tex, quote=False)
        return f"\\({tex}\\)" if kind == "inline" else f"\\[{tex}\\]"
    return re.sub(r"\x00MATH(\d+)\x00", repl, text)


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = text.replace("\\$", "$")          # markdown-escaped dollar -> literal
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  r'<a href="\2">\1</a>', text)
    return text


def img_tag(alt: str, src: str) -> str:
    path = os.path.join(ROOT, src)
    if os.path.exists(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        src = f"data:{mime};base64,{b64}"
    return (f'<figure><img alt="{html.escape(alt)}" src="{src}"/></figure>')


def render_table(rows: list[str]) -> str:
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]
    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    out = ["<table><thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def convert(md: str) -> str:
    # strip YAML front matter
    if md.startswith("---"):
        md = md.split("---", 2)[2]
    md = protect_math(md)
    lines = md.split("\n")
    out, i = [], 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # code fence
        if line.strip().startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i], quote=False)); i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        # horizontal rule
        if re.match(r"^---+\s*$", line):
            out.append("<hr/>"); i += 1; continue
        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2).strip())}</h{lvl}>")
            i += 1; continue
        # image (alone on a line)
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            out.append(img_tag(m.group(1), m.group(2))); i += 1; continue
        # table
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i+1]) and "-" in lines[i+1]:
            tbl = [line]; i += 1
            while i < n and "|" in lines[i] and lines[i].strip():
                tbl.append(lines[i]); i += 1
            out.append(render_table(tbl)); continue
        # blockquote
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i])); i += 1
            inner = convert_block_text("\n".join(buf))
            out.append(f"<blockquote>{inner}</blockquote>"); continue
        # unordered list
        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*[-*]\s+", "", lines[i]))); i += 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in buf) + "</ul>")
            continue
        # ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i]))); i += 1
            out.append("<ol>" + "".join(f"<li>{x}</li>" for x in buf) + "</ol>")
            continue
        # blank
        if not line.strip():
            i += 1; continue
        # paragraph (gather until blank / block start)
        buf = [line]; i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|```|>|\s*[-*]\s|\s*\d+\.\s|!\[)", lines[i]) \
                and not re.match(r"^---+\s*$", lines[i]):
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")
    return "\n".join(out)


def convert_block_text(t: str) -> str:
    # minimal: paragraphs only (used inside blockquotes)
    parts = [p for p in t.split("\n\n")]
    return "".join(f"<p>{inline(p.strip())}</p>" for p in parts if p.strip())


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Funded to Fail — A Double-Barrier Binary-Option Analysis</title>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},svg:{{fontCache:'global'}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>
<style>
:root{{--ink:#1a1a1a;--muted:#555;--accent:#1f3b73;--rule:#e2e2e2;}}
html{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
body{{max-width:820px;margin:0 auto;padding:48px 28px 96px;
 font:17px/1.62 Georgia,'Times New Roman',serif;color:var(--ink);background:#fff;}}
h1,h2,h3,h4{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 line-height:1.25;color:var(--accent);}}
h1{{font-size:2.0rem;margin:.2em 0 .1em;}}
h2{{font-size:1.45rem;margin:1.8em 0 .5em;border-bottom:2px solid var(--rule);padding-bottom:.2em;}}
h3{{font-size:1.18rem;margin:1.4em 0 .4em;}}
h4{{font-size:1.02rem;margin:1.2em 0 .3em;color:#333;}}
p{{margin:.6em 0;}}
a{{color:var(--accent);}}
hr{{border:0;border-top:1px solid var(--rule);margin:2em 0;}}
code{{font-family:SFMono-Regular,Menlo,Consolas,monospace;font-size:.86em;
 background:#f4f5f7;padding:.1em .35em;border-radius:4px;}}
pre{{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;overflow:auto;}}
pre code{{background:none;color:inherit;padding:0;}}
blockquote{{margin:1em 0;padding:.4em 1.1em;border-left:4px solid var(--accent);
 background:#f7f9fc;color:#222;font-style:italic;}}
table{{border-collapse:collapse;width:100%;margin:1.2em 0;font:14.5px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;}}
th,td{{border:1px solid var(--rule);padding:7px 10px;text-align:left;vertical-align:top;}}
thead th{{background:var(--accent);color:#fff;font-weight:600;}}
tbody tr:nth-child(even){{background:#f7f9fc;}}
figure{{margin:1.6em 0;text-align:center;}}
figure img{{max-width:100%;border:1px solid var(--rule);border-radius:6px;}}
em{{color:var(--muted);}}
@media print{{body{{font-size:11.5pt;padding:0;}}h2{{page-break-after:avoid;}}
 figure,table,pre,blockquote{{page-break-inside:avoid;}}}}
</style></head><body>
{body}
</body></html>"""


def main():
    with open(MD, encoding="utf-8") as fh:
        md = fh.read()
    body = restore_math(convert(md))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE.format(body=body))
    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({size:.0f} KB, figures embedded)")


if __name__ == "__main__":
    main()
