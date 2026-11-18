# © 2026 Computational Classnotes · CC BY-NC-SA 4.0
# Shared writer: Text-like cells get typeset subscripts (k_B) and superscripts (10^-6) as inline boxes.
import re
TOKEN = re.compile(r"(\u222b)_([^\s^]+)\^(\S+)|([A-Za-zΑ-ω]+)_([A-Za-z0-9]+)|([0-9A-Za-z\)Α-ω]+)\^(\{[^}]*\}|-?[0-9]+)")
def wl(s):
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + "".join(c if ord(c) < 128 else "\\:%04x" % ord(c) for c in s) + '"'
def textdata(t):
    parts, last = [], 0
    for m in TOKEN.finditer(t):
        if m.start() > last: parts.append(wl(t[last:m.start()]))
        g = m.groups()
        if g[0]:
            parts.append(f'Cell[BoxData[SubsuperscriptBox["\\[Integral]", {wl(g[1])}, {wl(g[2])}]]]')
        elif g[3]:
            parts.append(f'Cell[BoxData[SubscriptBox[{wl(g[3])}, {wl(g[4])}]]]')
        else:
            parts.append(f'Cell[BoxData[SuperscriptBox[{wl(g[5])}, {wl(g[6].strip("{}"))}]]]')
        last = m.end()
    if last < len(t): parts.append(wl(t[last:]))
    if len(parts) == 1 and parts[0].startswith('"'): return parts[0]
    return "TextData[{" + ", ".join(parts) + "}]"
def write(cells, path):
    out = []
    for s, t in cells:
        body = wl(t) if s == "Input" else textdata(t)
        out.append(f'Cell[{body}, "{s}"]')
    open(path, "w").write("Notebook[{\n" + ",\n".join(out) + '\n}, StyleDefinitions -> "Default.nb"]\n')
