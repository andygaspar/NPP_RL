"""
Remove the red change markers from the paper source, keeping the new text.
    \red{text}                         -> text
    \begin{redblock} ... \end{redblock} -> ...
The definitions of \red and redblock in the preamble are removed as well.
Usage: python remove_red.py [input] [output]   (default: tex_new -> tex_clean)
"""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else 'tex_new'
dst = sys.argv[2] if len(sys.argv) > 2 else 'tex_clean'

tex = open(src, encoding='utf-8').read()

# preamble definitions
for line in ['% changes with respect to the previous version are in red\n',
             '\\newcommand{\\red}[1]{\\textcolor{red}{#1}}\n',
             '\\newenvironment{redblock}{\\color{red}}{}\n']:
    tex = tex.replace(line, '')

# red blocks
tex = tex.replace('\\begin{redblock}\n', '').replace('\\end{redblock}\n', '')

# \red{...}, with balanced braces
out, i, tag = [], 0, '\\red{'
while True:
    j = tex.find(tag, i)
    if j < 0:
        out.append(tex[i:])
        break
    out.append(tex[i:j])
    k, depth = j + len(tag), 1
    while depth:
        if tex[k] == '\\':  # skip escaped characters such as \{ \} \%
            k += 2
            continue
        depth += {'{': 1, '}': -1}.get(tex[k], 0)
        k += 1
    out.append(tex[j + len(tag):k - 1])
    i = k
tex = ''.join(out)

assert '\\red{' not in tex and 'redblock' not in tex
open(dst, 'w', encoding='utf-8').write(tex)
print(f'{src} -> {dst}')
