"""Shared helpers for the step-by-step reports (Yang asked 2026-10-23: one detailed report per analysis step, every symbol
defined, every figure captioned with explicit axes and color scales).

A report is built in Python with Report(...): sections, paragraphs (with $...$ inline math), numbered display equations,
numbered figures (matplotlib, embedded as PNG and also saved to reports/figures/), tables, a symbol table and boxes.
save() renders the math with KaTeX (scripts/reports/katex, run with node, fonts embedded so the HTML works offline), writes
reports/<file>.html and prints reports/<file>.pdf with headless Chromium."""
import base64, io, json, os, re, subprocess, html as _h
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
B = '/mnt/project-files/apl_synapses/'
OUTDIR = B + 'reports/'
FIGDIR = OUTDIR + 'figures/'
OW = B + 'odor_weighted/'
CHROME = '/opt/pw-browsers/chromium'
DATE = '2026-10-23'

# ---------------------------------------------------------------- colors (same meaning in every report)
BLOCKED = '#eb6834'      # APL blocked: model "blocked", data TNT
INTACT = '#2a78d6'       # APL intact: model "intact", data TNT3c
GRAY = '#898781'
GREEN = '#1f9e6e'
PURPLE = '#8a5cc2'
INK, MUTED, GRID, NAN_C = '#0b0b0b', '#52514e', '#e1e0d9', '#d6d6d4'
DS_COLOR = {'hemibrain': '#eb6834', 'malecns': '#1f9e6e', 'flywire': '#2a78d6', 'banc': '#8a5cc2'}
DS_LABEL = {'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'flywire': 'FlyWire 783', 'banc': 'BANC 888'}
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#f7f7f5', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
SEQ_O = LinearSegmentedColormap.from_list('seq_orange', ['#f7f7f5', '#fbd9c8', '#f4a47f', '#eb6834', '#b4461c', '#6e2a10'])
DIV = LinearSegmentedColormap.from_list('div_blue_orange', ['#0d366b', '#3987e5', '#a9cdf5', '#efeee9', '#f6c0a6', '#eb6834', '#8f3412'])
for _c in (SEQ, SEQ_O, DIV): _c.set_bad(NAN_C)

REPORTS = {
    '00': ('00_overview_and_glossary', 'Overview, roadmap and glossary'),
    '01': ('01_connectome_synapse_tables', 'Connectome data: APL synapse tables and per-cell counts'),
    '02': ('02_claws_boutons_calyx', 'Claws, boutons, calyx geometry and the outside-ROI version'),
    '03': ('03_odor_weighted_maps', 'Odor-weighted maps of APL synapses'),
    '04': ('04_glomerulus_wiring', 'APL wiring per glomerulus: the connectome numbers the model uses'),
    '05': ('05_APL_block_model', 'The APL block model, built step by step'),
    '06': ('06_TNT_TNT3c_data', 'The TNT and TNT3c recordings: what is measured'),
    '07': ('07_model_vs_TNT', 'The model pinned to the recordings, and the comparison'),
    '08': ('08_why_feedback_is_uneven', 'Why APL feedback makes the effect of blocking uneven across odors'),
    '09': ('09_rank_order', 'Rank order of odor-pair similarity, with toy simulations'),
    '10': ('10_APL_geodesic_distances', 'Distances along APL between glomeruli'),
    '11': ('11_local_APL_model', 'A local, non-spiking APL in the KC model'),
    '12': ('12_APL_sub_arbor_compartments', 'APL confined to its calyx sub-arbors'),
}


def style():
    plt.rcParams.update({
        'font.family': 'Inter', 'font.size': 9, 'mathtext.fontset': 'cm', 'axes.titlesize': 9.5, 'axes.titleweight': 'semibold',
        'axes.titlelocation': 'left', 'axes.labelsize': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 8, 'figure.dpi': 100, 'savefig.dpi': 200,
        'lines.linewidth': 1.6, 'axes.titlepad': 6, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


style()


def cbar(fig, im, ax, label, **kw):
    cb = fig.colorbar(im, ax=ax, fraction=kw.pop('fraction', 0.046), pad=kw.pop('pad', 0.03), **kw)
    cb.set_label(label, fontsize=8.5, color=INK); cb.ax.tick_params(labelsize=7.5, colors=MUTED); cb.outline.set_visible(False)
    return cb


def tag(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left', color=INK)


def fmt(x, d=2, sign=False):
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    return s.replace('-', '−')


def pct(x, d=0):
    return f'{100 * x:.{d}f}%'


# ---------------------------------------------------------------- text markup
_MATH = re.compile(r'\$(.+?)\$')
_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ITAL = re.compile(r'(?<![\w*])_(.+?)_(?![\w*])')
_CODE = re.compile(r'`(.+?)`')
_LINK = re.compile(r'\[([^\]]+?)\]\(([^)\s]+?)\)')


class Report:
    def __init__(self, num, question, inputs=None, outputs=None, scripts=None):
        self.num = num; self.file, self.title = REPORTS[num]
        self.question = question; self.inputs, self.outputs, self.scripts = inputs or [], outputs or [], scripts or []
        self.parts, self.math = [], []
        self.n_sec = self.n_sub = self.n_fig = self.n_eq = self.n_tab = 0
        self.toc = []

    # -- markup
    def _m(self, tex, display=False):
        self.math.append((tex, display)); return f'\x00M{len(self.math) - 1}\x00'

    def t(self, s):
        """Text with $inline math$, **bold**, _italic_, `code` and [link](url); everything else is escaped."""
        keep = []
        def stash(x): keep.append(x); return f'\x01{len(keep) - 1}\x01'
        s = _MATH.sub(lambda m: stash(self._m(m.group(1))), s)
        s = _CODE.sub(lambda m: stash(f'<code>{_h.escape(m.group(1))}</code>'), s)
        s = _LINK.sub(lambda m: stash(f'<a href="{_h.escape(m.group(2))}">{_h.escape(m.group(1))}</a>'), s)
        s = _h.escape(s, quote=False)
        s = _BOLD.sub(r'<strong>\1</strong>', s)
        s = _ITAL.sub(r'<em>\1</em>', s)
        return re.sub('\x01(\\d+)\x01', lambda m: keep[int(m.group(1))], s)

    def ref(self, num, text=None):
        f, title = REPORTS[num]
        return f'[{text or f"Report {num}: {title}"}]({f}.html)'

    # -- blocks
    def html(self, s): self.parts.append(s)

    def section(self, title):
        self.n_sec += 1; self.n_sub = 0; a = f's{self.n_sec}'
        self.toc.append((a, f'{self.n_sec}  {title}', 0))
        self.parts.append(f'<h2 id="{a}"><span class="num">{self.n_sec}</span>{self.t(title)}</h2>')

    def sub(self, title):
        self.n_sub += 1; a = f's{self.n_sec}_{self.n_sub}'
        self.toc.append((a, f'{self.n_sec}.{self.n_sub}  {title}', 1))
        self.parts.append(f'<h3 id="{a}"><span class="num">{self.n_sec}.{self.n_sub}</span>{self.t(title)}</h3>')

    def p(self, s, cls=None):
        self.parts.append(f'<p{f" class={cls!r}" if cls else ""}>{self.t(s)}</p>')

    def ul(self, items, ordered=False):
        tagn = 'ol' if ordered else 'ul'
        def li(x):
            if isinstance(x, (list, tuple)):
                head, rest = x[0], x[1:]
                return f'<li>{self.t(head)}<{tagn}>' + ''.join(li(r) for r in (rest[0] if len(rest) == 1 and isinstance(rest[0], list) else rest)) + f'</{tagn}></li>'
            return f'<li>{self.t(x)}</li>'
        self.parts.append(f'<{tagn}>' + ''.join(li(x) for x in items) + f'</{tagn}>')

    def eq(self, tex, note=None):
        """Numbered display equation; returns its number."""
        self.n_eq += 1
        body = self._m(tex + f' \\tag{{{self.n_eq}}}', True)
        self.parts.append(f'<div class="eq" id="eq{self.n_eq}">{body}' + (f'<div class="eqnote">{self.t(note)}</div>' if note else '') + '</div>')
        return self.n_eq

    def fig(self, fig, caption, title, width=100):
        """Numbered figure: bold title then the caption. Saved to reports/figures/ and embedded. Returns its number."""
        self.n_fig += 1
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        data = buf.getvalue()
        os.makedirs(FIGDIR, exist_ok=True)
        with open(f'{FIGDIR}{self.file}_figure{self.n_fig}.png', 'wb') as fh: fh.write(data)
        b64 = base64.b64encode(data).decode()
        self.parts.append(f'<figure id="fig{self.n_fig}" style="width:{width}%"><img src="data:image/png;base64,{b64}" alt="Figure {self.n_fig}">'
                          f'<figcaption><strong>Figure {self.n_fig}. {self.t(title)}</strong> {self.t(caption)}</figcaption></figure>')
        return self.n_fig

    def table(self, header, rows, caption=None, title=None, cls='', num_cols=None):
        self.n_tab += 1
        num_cols = set(num_cols or [])
        th = ''.join(f'<th>{self.t(str(h))}</th>' for h in header)
        tr = ''.join('<tr>' + ''.join(f'<td{" class=n" if j in num_cols else ""}>{self.t(str(c))}</td>' for j, c in enumerate(r)) + '</tr>' for r in rows)
        cap = f'<caption><strong>Table {self.n_tab}. {self.t(title or "")}</strong> {self.t(caption or "")}</caption>' if (caption or title) else ''
        self.parts.append(f'<table class="{cls}">{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>')
        return self.n_tab

    def symbols(self, rows, title='Symbols used in this report'):
        """rows: (tex symbol, what it is, value / units / range)."""
        self.n_tab += 1
        tr = ''.join(f'<tr><td class="sym">{self._m(s)}</td><td>{self.t(m)}</td><td>{self.t(v)}</td></tr>' for s, m, v in rows)
        self.parts.append(f'<table class="symbols"><caption><strong>Table {self.n_tab}. {self.t(title)}.</strong></caption>'
                          f'<thead><tr><th>Symbol</th><th>What it stands for</th><th>Values, units, range</th></tr></thead><tbody>{tr}</tbody></table>')
        return self.n_tab

    def box(self, title, body, kind='plain'):
        inner = ''.join(f'<p>{self.t(b)}</p>' for b in body) if isinstance(body, (list, tuple)) else f'<p>{self.t(body)}</p>'
        self.parts.append(f'<div class="box {kind}"><div class="boxtitle">{self.t(title)}</div>{inner}</div>')

    # -- output
    def _render_math(self, s):
        if not self.math: return s
        r = subprocess.run(['node', os.path.join(HERE, 'katex', 'render.js')], input=json.dumps(self.math).encode(),
                           capture_output=True, check=True)
        out = json.loads(r.stdout)
        return re.sub('\x00M(\\d+)\x00', lambda m: out[int(m.group(1))], s)

    def _css(self):
        css = open(os.path.join(HERE, 'katex', 'katex.min.css')).read()
        def font(m):
            name = m.group(1)
            b64 = base64.b64encode(open(os.path.join(HERE, 'katex', 'fonts', name + '.woff2'), 'rb').read()).decode()
            return f'src:url(data:font/woff2;base64,{b64}) format("woff2")'
        css = re.sub(r'src:url\(fonts/([\w-]+)\.woff2\) format\("woff2"\),url\(fonts/[\w-]+\.woff\) format\("woff"\),url\(fonts/[\w-]+\.ttf\) format\("truetype"\)', font, css)
        return css + CSS

    def save(self):
        nav = []
        keys = list(REPORTS)
        i = keys.index(self.num)
        if i > 0: nav.append(f'<a href="{REPORTS[keys[i - 1]][0]}.html">← Report {keys[i - 1]}</a>')
        nav.append(f'<a href="{REPORTS["00"][0]}.html">All reports</a>')
        if i < len(keys) - 1: nav.append(f'<a href="{REPORTS[keys[i + 1]][0]}.html">Report {keys[i + 1]} →</a>')
        meta = ''
        for lab, xs in [('Inputs', self.inputs), ('Outputs', self.outputs), ('Scripts', self.scripts)]:
            if xs: meta += f'<div class="meta"><span class="metalab">{lab}</span><ul>' + ''.join(f'<li>{self.t(x)}</li>' for x in xs) + '</ul></div>'
        toc = ''.join(f'<li class="l{lvl}"><a href="#{a}">{_h.escape(t_)}</a></li>' for a, t_, lvl in self.toc)
        head = (f'<header><div class="kicker">APL synapses project · step-by-step reports · Report {self.num} · {DATE}</div>'
                f'<h1>{self.t(self.title)}</h1><div class="question"><span class="metalab">Question</span>{self.t(self.question)}</div>'
                f'{meta}<nav class="pn">{" · ".join(nav)}</nav></header>'
                + (f'<div class="toc"><div class="metalab">Contents</div><ul>{toc}</ul></div>' if self.toc else ''))
        body = self._render_math(head + '\n'.join(self.parts))
        doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
               f'<title>Report {self.num}: {_h.escape(self.title)}</title><style>{self._css()}</style></head>'
               f'<body><main>{body}<footer>{" · ".join(nav)}</footer></main></body></html>')
        os.makedirs(OUTDIR, exist_ok=True)
        hp, pp = OUTDIR + self.file + '.html', OUTDIR + self.file + '.pdf'
        with open(hp, 'w') as fh: fh.write(doc)
        subprocess.run([CHROME, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=20000',
                        f'--print-to-pdf={pp}', 'file://' + hp], capture_output=True, timeout=300)
        print(f'wrote {hp} ({os.path.getsize(hp) / 1e6:.1f} MB) and {pp} ({os.path.getsize(pp) / 1e6:.1f} MB): '
              f'{self.n_fig} figures, {self.n_eq} equations, {self.n_tab} tables')
        return hp, pp


CSS = r'''
:root { --ink:#0b0b0b; --muted:#52514e; --grid:#e1e0d9; --blue:#2a78d6; --orange:#eb6834; --bg:#ffffff; --panel:#f6f5f1; }
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0; color: var(--ink); background: var(--bg); font-family: Inter, "Helvetica Neue", Arial, system-ui, sans-serif;
       font-size: 10.5pt; line-height: 1.55; }
main { max-width: 860px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 21pt; line-height: 1.2; margin: 6px 0 12px; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 14.5pt; margin: 30px 0 8px; padding-top: 8px; border-top: 1.5px solid var(--ink); font-weight: 700; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 20px 0 6px; font-weight: 650; break-after: avoid; }
h2 .num, h3 .num { color: var(--muted); font-weight: 600; margin-right: 10px; }
p { margin: 6px 0 9px; }
ul, ol { margin: 4px 0 10px; padding-left: 22px; }
li { margin: 2px 0; }
code { font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: 8.6pt; background: var(--panel); padding: 0 3px; border-radius: 3px; word-break: break-all; }
a { color: #1c5cab; }
.kicker { font-size: 8.5pt; color: var(--muted); letter-spacing: 0.02em; text-transform: uppercase; }
.question { background: var(--panel); border-left: 3px solid var(--ink); padding: 8px 12px; margin: 8px 0 10px; }
.metalab { display: block; font-size: 8pt; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
.meta { margin: 6px 0; font-size: 9pt; }
.meta ul { margin: 0; }
nav.pn, footer { font-size: 9pt; color: var(--muted); margin-top: 8px; }
footer { margin-top: 36px; border-top: 1px solid var(--grid); padding-top: 8px; }
.toc { border: 1px solid var(--grid); border-radius: 6px; padding: 8px 14px; margin: 12px 0 6px; font-size: 9.2pt; }
.toc ul { list-style: none; padding-left: 0; margin: 0; columns: 2; column-gap: 28px; }
.toc li.l1 { padding-left: 18px; color: var(--muted); }
.toc a { color: var(--ink); text-decoration: none; }
.eq { margin: 10px 0 12px; break-inside: avoid; }
.eq .katex-display { margin: 4px 0; }
.eqnote { font-size: 9pt; color: var(--muted); text-align: center; margin-top: -2px; }
.katex { font-size: 1.08em; }
figure { margin: 16px auto 20px; break-inside: avoid; }
figure img { width: 100%; display: block; border: 1px solid #efeee9; }
figcaption { font-size: 9pt; line-height: 1.45; margin-top: 6px; color: #262624; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 9pt; break-inside: auto; }
caption { caption-side: top; text-align: left; font-size: 9pt; margin-bottom: 5px; color: #262624; }
th { text-align: left; border-bottom: 1.5px solid var(--ink); padding: 4px 6px; font-weight: 650; vertical-align: bottom; }
td { border-bottom: 1px solid var(--grid); padding: 4px 6px; vertical-align: top; }
td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr { break-inside: avoid; }
table.symbols td.sym { white-space: nowrap; width: 1%; padding-right: 14px; }
table.symbols td:nth-child(3) { width: 30%; color: #262624; }
.box { border-radius: 6px; padding: 8px 14px; margin: 12px 0 14px; break-inside: avoid; border: 1px solid var(--grid); }
.box p { margin: 4px 0; }
.box .boxtitle { font-weight: 700; font-size: 9.5pt; margin-bottom: 2px; }
.box.plain { background: #eef4fc; border-color: #cde2fb; }
.box.example { background: #f6f5f1; }
.box.caveat { background: #fdf1eb; border-color: #fbd9c8; }
.box.result { background: #eef8f3; border-color: #c4e6d6; }
@page { size: A4; margin: 14mm 13mm 15mm; }
@media print { main { max-width: none; padding: 0; } h2 { break-before: auto; } a { color: inherit; text-decoration: none; } }
@media (max-width: 600px) { .toc ul { columns: 1; } h1 { font-size: 17pt; } }
'''
