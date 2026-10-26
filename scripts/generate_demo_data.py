"""Generate the binary demo inputs for ProcureIQ.

Creates, inside data/demo/:
  - vendor_a_quote.pdf        formal PDF quotation (Vendor A, compliant)
  - vendor_c_handwritten.png  photo-style handwritten quote (Vendor C)

The text inputs (request email, WhatsApp chat, clarification reply) are
plain .txt files already committed in data/demo/.

Run from the repo root:
    python scripts/generate_demo_data.py
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
FONT_DIR = Path(__file__).resolve().parent / "fonts"


# ------------------------------------------------------------------ Vendor A
def make_vendor_a_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    s = getSampleStyleSheet()
    small = s["BodyText"].clone("small", fontSize=9, leading=12)
    story = [
        Paragraph("<b>KAVERI CABLES &amp; WIRES PVT LTD</b>", s["Title"]),
        Paragraph("Plot 14, SIDCO Industrial Estate, Guindy, Chennai 600032 &nbsp;|&nbsp; "
                  "GSTIN 33AAACK0000A1Z5 (demo)", small),
        Spacer(1, 8),
        Paragraph("<b>QUOTATION</b>", s["Heading2"]),
        Table([
            ["Quotation No.", "KCW/Q/2026/552", "Date", "20-10-2026"],
            ["To", "Sunrise Manufacturing, Ambattur", "Ref", "Block C wiring enquiry"],
        ], colWidths=[28 * mm, 70 * mm, 18 * mm, 50 * mm],
            style=TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                              ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                              ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold")])),
        Spacer(1, 10),
    ]

    desc = Paragraph(
        "KaveriFlex FR 2.5 sq mm single core PVC insulated (Type A) wire, "
        "electrolytic grade copper conductor, Class 2 stranded, 1.1 kV grade, "
        "conforming to IS 694:2010. ISI marked, licence CM/L-4100012345. "
        "FR compound: oxygen index 30, temperature index 255 &deg;C. 90 m coil.", small)
    items = [
        ["#", "Description", "Qty", "Unit", "Rate (INR)", "Amount (INR)"],
        ["1", desc, "20", "Coil", "21,000.00", "4,20,000.00"],
        ["", "", "", "", "Sub-total", "4,20,000.00"],
        ["", "", "", "", "GST @ 18%", "75,600.00"],
        ["", "", "", "", "Grand total", "4,95,600.00"],
    ]
    t = Table(items, colWidths=[8 * mm, 82 * mm, 12 * mm, 14 * mm, 26 * mm, 30 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, 1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (4, 2), (5, -1), "Helvetica-Bold"),
        ("LINEABOVE", (4, 2), (5, 2), 0.5, colors.grey),
    ]))
    story += [t, Spacer(1, 14), Paragraph("<b>Terms &amp; conditions</b>", s["Heading4"])]
    for line in [
        "1. Delivery: within 10 days from receipt of PO, door delivery to Ambattur.",
        "2. Warranty: 3 years against manufacturing defects.",
        "3. Payment: 30 days from date of invoice.",
        "4. Validity of offer: 15 days.",
        "5. Test certificate and copy of BIS licence will accompany the supply.",
    ]:
        story.append(Paragraph(line, small))
    story += [Spacer(1, 20), Paragraph("For Kaveri Cables &amp; Wires Pvt Ltd<br/><br/>"
                                       "Authorised Signatory", small)]
    doc.build(story)


# ------------------------------------------------------------------ Vendor C
C_LINES = [
    ("ANNA NAGAR WIRE HOUSE", 1.25),
    ("Quotation: AWH/2026/118      Date: 20/10/26", 1.0),
    ("To: Sunrise Mfg, Ambattur - Block C", 1.0),
    ("", 0.6),
    ("AnnaWire 2.5 sq mm 1C copper wire", 1.0),
    ("1100V PVC, as per IS 694, FR grade", 1.0),
    ("ISI  CM/L-4100067890", 1.0),
    ("Class 2 stranded, 90 m coil", 1.0),
    ("", 0.6),
    ("Qty : 20 coils", 1.0),
    ("Rate : 22,750 / coil", 1.0),          # rendered faded -> low-confidence field
    ("Total : 4,55,000 + GST 18%", 1.0),
    ("", 0.6),
    ("Delivery : 12 days from PO", 1.0),
    ("Warranty : 2 yrs", 1.0),
    ("", 0.6),
    ("- Meenakshi", 1.0),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    hand = FONT_DIR / "Kalam-Regular.ttf"
    if hand.exists():
        return ImageFont.truetype(str(hand), size)
    try:  # fallback keeps the script runnable without the bundled font
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_vendor_c_png(path: Path, seed: int = 7) -> None:
    rnd = random.Random(seed)
    W, H = 1240, 1560
    img = Image.new("RGB", (W, H), (247, 243, 228))  # off-white notepad
    d = ImageDraw.Draw(img)

    # ruled lines + margin
    for y in range(150, H - 60, 76):
        d.line([(60, y), (W - 60, y)], fill=(186, 205, 228), width=2)
    d.line([(150, 40), (150, H - 40)], fill=(226, 160, 160), width=2)

    y = 92
    for text, scale in C_LINES:
        if text:
            f = _font(int(52 * scale))
            x = 175 + rnd.randint(-6, 10)
            ink = (28, 42, 110)
            if text.startswith("Rate"):
                ink = (120, 130, 175)  # faded ink on the price line
            # draw word by word with slight jitter for a hand-written feel
            for word in text.split(" "):
                if not word:
                    x += 18
                    continue
                d.text((x, y + rnd.randint(-3, 3)), word, font=f, fill=ink)
                x += int(d.textlength(word + " ", font=f))
        y += int(76 * (scale if text else 0.9))

    # photo feel: slight rotation, blur, vignette
    img = img.rotate(-1.6, resample=Image.BICUBIC, expand=False, fillcolor=(120, 112, 98))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for i in range(40):
        sd.rectangle([i * 4, i * 4, W - i * 4, H - i * 4], outline=int(70 - i * 1.7))
    img = Image.composite(Image.new("RGB", (W, H), (60, 55, 45)), img, shade)
    img.save(path, quality=90)


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    make_vendor_a_pdf(DEMO / "vendor_a_quote.pdf")
    make_vendor_c_png(DEMO / "vendor_c_handwritten.png")
    print("Generated:", *(p.name for p in sorted(DEMO.iterdir())), sep="\n  ")


if __name__ == "__main__":
    main()
