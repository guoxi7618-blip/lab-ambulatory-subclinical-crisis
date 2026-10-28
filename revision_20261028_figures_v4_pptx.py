"""Assemble the v4 figures in PowerPoint (python-pptx).
Schematic content (Figures 1, 2A-B, S1, S2), panel labels and legends are native, editable shapes;
data panels are the 600-dpi images produced by revision_20261028_figures_v4_panels.py.
Outputs: Figures/v4/Psychophysiology_figures_v4.pptx (all figures) and one single-slide deck per
figure in Figures/v4/slides/ for export (export_figures_v4.ps1).
"""
from pathlib import Path
import sys, copy
from pptx import Presentation
from pptx.util import Mm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parent
V4 = ROOT / 'Figures' / 'v4'; PANELS = V4 / 'panels'; SLIDES = V4 / 'slides'
SLIDES.mkdir(parents=True, exist_ok=True)
W = 180
CTRL, SUB = '#2F6A9F', '#D07A2B'
CORE, EXT = '#262626', '#8A8A8A'
INK, DARK, MIDG, LIGHT, PALE = '#1A1A1A', '#4D4D4D', '#BFBFBF', '#E6E6E6', '#F4F4F4'
STRESS, RECOV = '#F3D2B3', '#CFDDEB'
FONT = 'Arial'


def rgb(h): return RGBColor.from_string(h.lstrip('#'))


class Slide:
    def __init__(self, prs, name, height):
        self.s = prs.slides.add_slide(prs.slide_layouts[6]); self.name = name; self.h = height; self.n = 0

    def _name(self, shp, kind):
        self.n += 1; shp.name = f'{self.name}_{self.n:03d}_{kind}'; return shp

    def text(self, x, y, w, h, s, size=7, bold=False, color=INK, align='left', anchor='middle', italic=False, wrap=True):
        tb = self.s.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h)); tf = tb.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = wrap; tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = {'top': MSO_ANCHOR.TOP, 'middle': MSO_ANCHOR.MIDDLE, 'bottom': MSO_ANCHOR.BOTTOM}[anchor]
        for i, line in enumerate(s.split('\n')):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {'left': PP_ALIGN.LEFT, 'center': PP_ALIGN.CENTER, 'right': PP_ALIGN.RIGHT}[align]
            p.line_spacing = 1.0; p.space_before = p.space_after = Pt(0)
            # **bold** segments
            parts = line.split('**')
            for j, seg in enumerate(parts):
                if not seg: continue
                r = p.add_run(); r.text = seg; f = r.font
                f.name = FONT; f.size = Pt(size); f.bold = bold or (j % 2 == 1); f.italic = italic; f.color.rgb = rgb(color)
        return self._name(tb, 'text')

    def panel(self, x, y, letter, title, w=90):
        return self.text(x, y, w, 5, f'{letter} | {title}', 8.5, bold=True, anchor='middle')

    def rect(self, x, y, w, h, fill=LIGHT, line=None, lw=0.6, shape=MSO_SHAPE.RECTANGLE, dash=False):
        r = self.s.shapes.add_shape(shape, Mm(x), Mm(y), Mm(w), Mm(h))
        if fill is None: r.fill.background()
        else: r.fill.solid(); r.fill.fore_color.rgb = rgb(fill)
        if line is None: r.line.fill.background()
        else:
            r.line.color.rgb = rgb(line); r.line.width = Pt(lw)
            if dash:
                from pptx.enum.dml import MSO_LINE_DASH_STYLE
                r.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        r.shadow.inherit = False
        if r.has_text_frame: r.text_frame.text = ''
        return self._name(r, 'rect')

    def box(self, x, y, w, h, title, body='', fill=LIGHT, line=None, tcolor=INK, tsize=7.5, bsize=6.8, header=None):
        """Box with optional dark header bar (Psychophysiology schematic style)."""
        if header:
            self.rect(x, y, w, 5.5, header)
            self.text(x + 1, y, w - 2, 5.5, title, tsize, True, '#FFFFFF', 'center')
            self.rect(x, y + 5.5, w, h - 5.5, fill, line)
            if body: self.text(x + 1.5, y + 6.5, w - 3, h - 7.5, body, bsize, align='center', anchor='middle')
        else:
            self.rect(x, y, w, h, fill, line)
            if body:
                self.text(x + 1, y + 1, w - 2, 4.5, title, tsize, True, tcolor, 'center')
                self.text(x + 1.5, y + 5.5, w - 3, h - 6.5, body, bsize, align='center', anchor='middle')
            else:
                self.text(x + 1, y, w - 2, h, title, tsize, True, tcolor, 'center')

    def line(self, x1, y1, x2, y2, color=DARK, lw=0.75, arrow=False, dash=False):
        c = self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
        c.line.color.rgb = rgb(color); c.line.width = Pt(lw)
        ln = c.line._get_or_add_ln()
        if dash:
            pd_ = etree.SubElement(ln, qn('a:prstDash')); pd_.set('val', 'dash')
        if arrow:
            te = etree.SubElement(ln, qn('a:tailEnd')); te.set('type', 'triangle'); te.set('w', 'med'); te.set('len', 'med')
        return self._name(c, 'arrow' if arrow else 'line')

    def marker(self, x, y, r, color, shape=MSO_SHAPE.OVAL, fill=True):
        return self.rect(x - r, y - r, 2 * r, 2 * r, color if fill else '#FFFFFF', color, 0.6, shape)

    def pic(self, x, y, name, w, h):
        p = self.s.shapes.add_picture(str(PANELS / f'{name}.png'), Mm(x), Mm(y), Mm(w), Mm(h))
        p.name = f'{self.name}_panel_{name}'; return p

    def legend(self, x, y, items, size=7, gap=4):
        """items: (label, color, kind) with kind in line/circle/square/open_circle/open_square."""
        cx = x
        for lab, col, kind in items:
            if kind == 'line': self.line(cx, y, cx + 5, y, col, 1.2); dx = 6
            else:
                shp = {'square': MSO_SHAPE.RECTANGLE, 'triangle': MSO_SHAPE.ISOSCELES_TRIANGLE, 'diamond': MSO_SHAPE.DIAMOND}.get(kind.replace('open_', ''), MSO_SHAPE.OVAL)
                self.marker(cx + 1.2, y, 1.0, col, shp, not kind.startswith('open')); dx = 3.2
            tw = 1.45 * len(lab) * size / 7 + 1
            self.text(cx + dx, y - 2.2, tw, 4.4, lab, size, wrap=False)
            cx += dx + tw + gap


# ------------------------------------------------------------------ figures
def figure1(prs):
    H = 109; s = Slide(prs, 'Figure1', H)
    s.panel(0, 1, 'A', 'Two-semester design', 120)
    lanes = [(9, 'Core cohort', 'Autumn 2025 · development\nOct 24, 2025 – Jan 17, 2026\nn = 203 (102 subclinical, 101 control)', CORE),
             (43, 'Extension cohort', 'Spring 2026 · temporal validation\nFeb 22 – May 16, 2026\nn = 198 (95 subclinical, 103 control)', EXT)]
    steps = [('Online screening', 'PHQ-9, GAD-7'), ('Counselor evaluation', 'Day −3 to −1\nreference standard, C-SSRS'),
             ('Day 0 laboratory', 'Stress task (MIST)\nEEG, ECG, EDA, wrist'), ('Days 1–30', 'Wrist monitoring\n4 EMA prompts + diary/day'),
             ('Day 31', 'Laboratory retest\ncounselor re-evaluation')]
    for y, title, body, col in lanes:
        s.rect(0, y, 45, 24, '#FFFFFF', col, 0.9)
        s.text(1.5, y + 1, 42, 5, title, 8, True, col, 'left')
        s.text(1.5, y + 6.5, 42, 16, body, 6.8, anchor='top')
        for i, (t, b) in enumerate(steps):
            x = 50 + i * 26.2
            if i == 4 and title.startswith('Extension'):
                s.rect(x, y + 2, 23.5, 20, None, MIDG, 0.6, dash=True)
                s.text(x, y + 2, 23.5, 20, 'Not scheduled', 6.8, color='#8C8C8C', align='center'); continue
            s.box(x, y + 2, 23.5, 20, t, b, PALE, col, tsize=7, bsize=6.3)
            if i: s.line(x - 2.7, y + 12, x - 0.2, y + 12, DARK, 0.75, True)
        s.line(45, y + 12, 49.8, y + 12, DARK, 0.75, True)
    s.line(22.5, 33.2, 22.5, 42.8, DARK, 0.9, True)
    s.text(24.5, 34.2, 150, 7.5, 'Models, preprocessing and decision thresholds developed in the core cohort were frozen (Feb 26, 2026)\nand applied unchanged to the extension cohort', 6.6, color=DARK, anchor='middle')
    s.panel(0, 71, 'B', 'Research questions and levels of analysis', 120)
    cols = [('H1  Laboratory markers', 'Participant level, Day 0\n14 candidate markers, core cohort\n5 core-selected markers\nreplicated in extension\nFigures 3–4; Table 3'),
            ('H2  Laboratory–field link', 'Prompt level, 30-min windows\n32,891 prompts, 401 participants\nStressor × laboratory\nHR reactivity\nFigure 5A'),
            ('H3  Identification', 'Participant level, extension\nn = 198, archived predictions\nAUC vs. SDS/SAS,\nablations, simpler models\nFigures 5B and 6; Table 4'),
            ('H4  Ecological performance', 'Day level, extension\n4,947 person-days\nScenarios, examination weeks,\nwithin-person discrimination\nFigure 7; Table 5')]
    for i, (t, b) in enumerate(cols):
        s.box(i * 45.3, 78, 43.5, 30, t, b, PALE, None, header=DARK, tsize=7.2, bsize=6.6)
    return s


def figure2(prs):
    H = 125; s = Slide(prs, 'Figure2', H)
    s.panel(0, 1, 'A', 'Laboratory session (Day 0)', 120)
    x0, x1, T = 31, 177, 41.0
    tx = lambda t: x0 + (x1 - x0) * t / T
    segs = [(0, 5, 'Adaptation', LIGHT), (5, 13, 'Baseline\nEC + EO', LIGHT), (13, 16, 'Control', MIDG),
            (16, 27, 'Stress\nS1–S3', STRESS), (27, 32, 'Recovery\nR1', RECOV), (32, 37, 'Recovery\nR2', RECOV), (37, 41, 'Post\nEC + EO', LIGHT)]
    s.text(0, 9, 29, 10, 'Phase', 7, True, align='right')
    for a, b, lab, col in segs:
        s.rect(tx(a), 9, tx(b) - tx(a), 10, col, '#FFFFFF', 0.8)
        s.text(tx(a), 9, tx(b) - tx(a), 10, lab, 6.5, align='center')
    s.text(0, 21, 29, 5, 'Ratings', 7, True, align='right')
    for t in (13, 27, 37):
        s.marker(tx(t), 23.5, 1.1, CORE, MSO_SHAPE.ISOSCELES_TRIANGLE)
    for t in (16, 37):
        s.marker(tx(t), 27.0, 1.1, SUB, MSO_SHAPE.DIAMOND)
    s.legend(98, 3.5, [('VAS stress + STAI-6', CORE, 'triangle'), ('PANAS (PA, NA)', SUB, 'diamond')], 6.6, 5)
    rows = [('EEG', '32 semi-dry channels, 500 Hz'), ('ECG', 'modified lead II, 1,000 Hz'),
            ('EDA', 'palmar, constant voltage, 250 Hz'), ('Wrist', 'EmbracePlus, concurrent')]
    for i, (a, b) in enumerate(rows):
        y = 33 + i * 5.2
        s.text(0, y - 2.4, 29, 4.8, f'**{a}**', 7, align='right')
        s.line(x0, y, x1, y, DARK if i < 3 else EXT, 1.6)
        s.text(tx(0.3), y - 4.4, 70, 3.6, b, 6.2, color=DARK, wrap=False)
    s.line(x0, 55, x1, 55, INK, 0.6)
    for t in range(0, 41, 5):
        s.line(tx(t), 55, tx(t), 56.2, INK, 0.6); s.text(tx(t) - 4, 56.6, 8, 3.6, str(t), 6.8, align='center')
    s.text(x0, 60, x1 - x0, 4, 'Time from adaptation onset (min)', 7, align='center')
    # B daily ambulatory schedule
    s.panel(0, 67, 'B', 'Ambulatory day (Days 1–30)', 100)
    a0, a1 = 12, 106
    hx = lambda h: a0 + (a1 - a0) * (h - 8) / 14
    s.text(0, 76, 11.5, 6, 'EMA', 7, True, align='right')
    for i, (h0, h1) in enumerate([(9, 11), (12, 14), (15, 17), (19, 21)]):
        s.rect(hx(h0), 76, hx(h1) - hx(h0), 6, RECOV, None)
        s.text(hx(h0), 76, hx(h1) - hx(h0), 6, f'Prompt {i + 1}', 6.5, align='center')
    s.marker(hx(21.5), 79, 1.4, SUB, MSO_SHAPE.DIAMOND); s.text(hx(21.5) - 6, 82.4, 12, 3.5, 'Diary', 6.5, align='center')
    # pre-prompt window example
    s.rect(hx(9.67), 86, hx(10.17) - hx(9.67), 4, STRESS, None)
    s.line(hx(10.17), 82, hx(10.17), 91, CORE, 0.7)
    s.text(hx(10.4), 85.5, 60, 5, 'Completed prompt + preceding 30-min physiology window', 6.4, color=DARK, wrap=False)
    s.text(0, 93, 11.5, 5, 'Wrist', 7, True, align='right')
    s.line(a0, 95.5, a1, 95.5, EXT, 1.6)
    s.text(a0 + 1, 96.6, 90, 3.8, 'Continuous wear including sleep; valid day ≥ 16 h', 6.4, color=DARK, wrap=False)
    s.line(a0, 102, a1, 102, INK, 0.6)
    for h in range(8, 23, 2):
        s.line(hx(h), 102, hx(h), 103.2, INK, 0.6); s.text(hx(h) - 5, 103.6, 10, 3.6, f'{h:02d}:00', 6.6, align='center')
    s.text(a0, 108, a1 - a0, 4, 'Time of day', 7, align='center')
    s.text(0, 114, 108, 12, 'Prompts: stress, positive and negative affect, stressor occurrence and type, activity, location.\nDiary: coping, rumination, passive suicidal ideation. Reminder after 15 min.', 6.4, color=DARK, anchor='top')
    s.panel(112, 67, 'C', 'Stress manipulation check', 68)
    s.pic(112, 74, 'F2C_vas', 66, 50)
    return s


def figure3(prs):
    H = 142; s = Slide(prs, 'Figure3', H)
    for i, (L, t, n) in enumerate([('A', 'Heart rate', 'F3A_hr'), ('B', 'Vagally mediated HRV', 'F3B_lnrmssd'), ('C', 'Skin conductance level', 'F3C_scl')]):
        s.panel(i * 61, 1, L, t, 60); s.pic(i * 61, 7, n, 58, 50)
    s.legend(46, 60, [('Control (n = 101)', CTRL, 'circle'), ('Subclinical (n = 102)', SUB, 'circle')], 7, 6)
    s.text(118, 57.8, 60, 4.4, 'Core cohort, mean ± SEM', 6.6, color=DARK, wrap=False)
    s.panel(0, 66, 'D', 'Candidate markers: core selection and replication', 100)
    s.legend(30, 73.5, [('Core', CORE, 'circle'), ('Extension', EXT, 'square')], 7, 5)
    s.text(70, 71.3, 30, 4.4, 'Filled: FDR p < 0.05', 6.6, color=DARK, wrap=False)
    s.pic(0, 76, 'F3D_forest', 98, 66)
    s.panel(102, 66, 'E', 'Replicated marker: vagal withdrawal', 78)
    s.pic(102, 72, 'F3E_reactivity', 78, 70)
    return s


def figure4(prs):
    H = 146; s = Slide(prs, 'Figure4', H)
    s.panel(0, 1, 'A', 'Aperiodic exponent, core cohort (97 control, 100 subclinical)', 170)
    s.pic(0, 7, 'F4A_core', 180, 40)
    s.text(140, 42, 40, 4, '○ max-T FWER p < 0.05', 6.6, color=DARK, wrap=False)
    s.panel(0, 49, 'B', 'Aperiodic exponent, extension cohort (97 control, 92 subclinical)', 170)
    s.pic(0, 55, 'F4B_ext', 180, 40)
    s.panel(0, 97, 'C', 'Oscillatory power, core cohort (subclinical − control)', 110)
    s.pic(0, 103, 'F4C_bands', 110, 42)
    s.panel(112, 97, 'D', 'Channel-mean exponent', 68)
    s.pic(112, 102, 'F4D_exponent', 68, 44)
    return s


def figure5(prs):
    H = 77; s = Slide(prs, 'Figure5', H)
    s.panel(0, 1, 'A', 'Laboratory–field correspondence', 90); s.pic(0, 8, 'F5A_scatter', 86, 68)
    s.panel(94, 1, 'B', 'Laboratory calibration and modality ablations', 86); s.pic(94, 8, 'F5B_ablation', 86, 68)
    return s


def figure6(prs):
    H = 133; s = Slide(prs, 'Figure6', H)
    for i, (L, t, n) in enumerate([('A', 'Discrimination', 'F6A_roc'), ('B', 'Calibration', 'F6B_calibration'), ('C', 'Crisis presentation', 'F6D_subgroups')]):
        s.panel(i * 61, 1, L, t, 60); s.pic(i * 61, 7, n, 58, 52)
    s.panel(0, 62, 'D', 'Extension-cohort AUC across models (n = 198)', 170)
    s.pic(0, 67, 'F6C_forest', 180, 66)
    return s


def figure7(prs):
    H = 136; s = Slide(prs, 'Figure7', H)
    s.panel(0, 1, 'A', 'Daily accuracy by scenario (person-days)', 90); s.pic(0, 7, 'F7A_scenarios', 86, 58)
    s.panel(90, 1, 'B', 'Exam weeks', 44); s.pic(90, 7, 'F7B_exam', 40, 58)
    s.panel(134, 1, 'C', 'Adjusted GEE', 46); s.pic(134, 7, 'F7C_gee', 44, 58)
    s.panel(0, 69, 'D', 'Pooled vs. within-person AUC', 90); s.pic(0, 76, 'F7D_within', 86, 60)
    s.panel(92, 69, 'E', 'Within-person AUC by participant (n = 198)', 88); s.pic(92, 76, 'F7E_individual', 88, 60)
    return s


def figureS1(prs):
    H = 150; s = Slide(prs, 'FigureS1', H)
    s.panel(0, 1, 'A', 'Participant flow across both semesters', 120)
    def fb(x, y, w, h, t, b, line=DARK): s.box(x, y, w, h, t, b, '#FFFFFF', line, tsize=7.2, bsize=6.8)
    fb(60, 9, 60, 12, 'Screened online', '2,330 (core 1,180; extension 1,150)')
    s.line(90, 21, 90, 27, DARK, 0.75, True)
    fb(60, 27, 60, 12, 'Invited to evaluation', '1,620 (810; 810)')
    s.line(90, 39, 90, 45, DARK, 0.75, True)
    s.line(90, 24, 127, 24, DARK, 0.6, True); s.text(128, 21, 52, 6, 'Not invited: 710 (370; 340)', 6.8)
    s.line(90, 42, 127, 42, DARK, 0.6, True); s.text(128, 39, 52, 6, 'Did not attend: 250 (130; 120)', 6.8)
    fb(60, 45, 60, 12, 'Counselor evaluation', '1,370 (680; 690)')
    s.line(90, 57, 90, 61, DARK, 0.75); s.line(40, 61, 130, 61, DARK, 0.75)
    for x, lab, col, ev, ex, enr, lost, an in [
            (10, 'Core cohort', CORE, '680', 'Probable mental disorder, referred: 44\nOther eligibility exclusions: 30\nDeclined after evaluation: 21\nEligible, quota filled: 355', '230 (115 / 115)', '27', '203 (102 / 101)'),
            (100, 'Extension cohort', EXT, '690', 'Probable mental disorder, referred: 36\nOther eligibility exclusions: 32\nDeclined after evaluation: 18\nEligible, quota filled: 374', '230 (115 / 115)', '32', '198 (95 / 103)')]:
        cx = x + 30
        s.line(cx, 61, cx, 65, DARK, 0.75, True)
        s.text(x, 65, 60, 5, lab, 8, True, col, 'center')
        s.rect(x, 71, 60, 22, PALE, None); s.text(x + 2, 71.5, 56, 21, f'**Not enrolled ({int(ev) - 230})**\n' + ex, 6.6, anchor='middle')
        s.line(cx, 93, cx, 98, DARK, 0.75, True)
        fb(x, 98, 60, 12, 'Enrolled', enr + ' subclinical / control', col)
        s.line(cx, 110, cx, 126, DARK, 0.75, True)
        s.text(cx + 2.5, 113, 40, 9, f'Not analyzable: {lost}\n(reasons overlap; see note)', 6.6, color=DARK, align='left')
        fb(x, 126, 60, 12, 'Analyzed', an + ' subclinical / control', col)
    s.text(0, 141, 180, 9, 'Post-enrollment exclusion reasons (overlapping): withdrawal before laboratory recording (2), < 20 valid wearable days (54, including 5 without wearable records),\n< 50% of semi-random EMA prompts completed (13); 59 unique participants in total. Four tracking withdrawals met the data thresholds and were retained.', 6.3, color=DARK, anchor='top')
    return s


def figureS2(prs):
    H = 122; s = Slide(prs, 'FigureS2', H)
    s.panel(0, 1, 'A', 'Multiscale attention-based fusion (MSAF) model', 140)
    ins = [(0, 'Laboratory EEG', '32 channel tokens\n8 features + channel embedding'),
           (45.5, 'Laboratory autonomic', '9 features\n(HR, lnRMSSD, lnHF, SCL ...)'),
           (91, 'Wrist windows', '≤ 288 five-minute windows/day\nHR, lnRMSSD, SCL, SCR, temp., ENMO'),
           (136.5, 'EMA', 'Daily summaries (9)\nstress, affect, stressors')]
    for x, t, b in ins:
        s.box(x, 9, 43.5, 17, t, b, PALE, None, header=DARK, tsize=7, bsize=6.3)
    enc = [(0, '2-layer Transformer'), (45.5, 'MLP 9 → 32 → 32'), (91, '2 × Conv1D + attention pooling')]
    for x, t in enc:
        s.line(x + 21.75, 26, x + 21.75, 30, DARK, 0.75, True)
        s.box(x, 30, 43.5, 8, t, '', '#FFFFFF', DARK, tsize=6.8)
    s.line(158.25, 26, 158.25, 47, DARK, 0.75)
    s.line(158.25, 47, 134.7, 47, DARK, 0.75, True)
    s.line(112.75, 38, 112.75, 42, DARK, 0.75, True)
    s.box(91, 42, 43.5, 10, 'Day tokens', '', '#FFFFFF', DARK, tsize=6.8)
    s.text(91, 52.3, 43.5, 4, '+ 6 wrist summaries, EMA, calendar', 6.0, color=DARK, align='center')
    s.line(112.75, 56.5, 112.75, 60, DARK, 0.75, True)
    s.box(91, 60, 43.5, 9, '30-day Transformer (2 layers)', '', '#FFFFFF', DARK, tsize=6.8)
    # laboratory conditioning
    s.rect(45.5, 44, 38, 17, '#FFFFFF', SUB, 0.9)
    s.text(46.5, 44.5, 36, 16, '**Laboratory calibration**\nanchored normalization\n(reference level, reactivity,\nwrist–laboratory mapping)\n+ FiLM scale and shift', 6.3, color=INK, align='center')
    s.line(67.25, 38, 67.25, 43.8, SUB, 0.9, True)
    s.line(83.5, 52, 90.8, 47, SUB, 0.9, True); s.line(83.5, 55, 90.8, 64.5, SUB, 0.9, True)
    # fusion
    s.box(10, 79, 160, 12, 'Cross-modal fusion Transformer', '[CLS] + EEG, autonomic, ecological and EMA tokens; learned mask tokens for missing modalities', PALE, DARK, tsize=7.2, bsize=6.4)
    for x in (21.75, 67.25, 112.75, 158.25):
        pass
    s.line(21.75, 38, 21.75, 78.8, DARK, 0.75, True)
    s.line(67.25, 61, 67.25, 78.8, DARK, 0.75, True)
    s.line(112.75, 69, 112.75, 78.8, DARK, 0.75, True)
    s.line(158.25, 47, 158.25, 78.8, DARK, 0.75, True)
    s.line(55, 91, 55, 97, DARK, 0.75, True); s.line(125, 91, 125, 97, DARK, 0.75, True)
    s.box(20, 97, 70, 12, 'Classification head', 'Subclinical crisis vs. control (weighted BCE)', '#FFFFFF', CTRL, tsize=7.2, bsize=6.4)
    s.box(90 + 5, 97, 65, 12, 'Auxiliary head', 'Counselor severity, 0–10 (MSE, weight 0.3)', '#FFFFFF', SUB, tsize=7.2, bsize=6.4)
    s.text(0, 112, 180, 9, 'Width 32; 57,507 trainable parameters; masked-window pretraining (6 epochs); AdamW; five participant-stratified folds × five seeds;\nthreshold maximizing specificity at sensitivity ≥ 0.80 in the core cohort.', 6.3, color=DARK, anchor='top', align='center')
    return s


BUILDERS = [('Figure1', figure1, 109), ('Figure2', figure2, 125), ('Figure3', figure3, 142), ('Figure4', figure4, 146),
            ('Figure5', figure5, 77), ('Figure6', figure6, 133), ('Figure7', figure7, 136),
            ('FigureS1', figureS1, 150), ('FigureS2', figureS2, 122)]


def new_prs(h):
    prs = Presentation(); prs.slide_width = Mm(W); prs.slide_height = Mm(h); return prs


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    # one deck per figure (exact slide height), plus a combined deck (max height, figure top-aligned)
    hmax = max(h for *_, h in BUILDERS)
    combo = new_prs(hmax)
    for name, fn, h in BUILDERS:
        prs = new_prs(h); fn(prs); prs.core_properties.title = name
        prs.save(SLIDES / f'{name}.pptx')
        fn(combo)
        combo.slides[-1].notes_slide.notes_text_frame.text = f'{name}: slide height for export = {h} mm (width 180 mm).'
    combo.core_properties.title = 'Psychophysiology figures v4 (editable)'
    combo.save(V4 / 'Psychophysiology_figures_v4_editable.pptx')
    print('saved', len(BUILDERS), 'figures')


if __name__ == '__main__':
    main()
