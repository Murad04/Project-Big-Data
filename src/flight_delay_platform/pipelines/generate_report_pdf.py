from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


def parse_markdown_lines(md_text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            items.append(("blank", ""))
        elif line.startswith("# "):
            items.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            items.append(("h2", line[3:].strip()))
        elif line.startswith("### "):
            items.append(("h3", line[4:].strip()))
        elif line.startswith("- "):
            items.append(("bullet", line[2:].strip()))
        elif line[0].isdigit() and ". " in line:
            head, _, tail = line.partition(". ")
            if head.isdigit():
                items.append(("number", tail.strip()))
            else:
                items.append(("p", line.strip()))
        else:
            items.append(("p", line.strip()))
    return items


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    parsed = parse_markdown_lines(text)

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=12,
        textColor=colors.HexColor("#0f172a"),
    )
    style_h2 = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#1e293b"),
    )
    style_h3 = ParagraphStyle(
        "H3Style",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#334155"),
    )
    style_p = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Flight Delay Prediction Platform Report",
        author="Project Team",
    )

    flow = []
    bullet_buffer: list[str] = []
    number_buffer: list[str] = []

    def flush_bullets() -> None:
        nonlocal bullet_buffer
        if bullet_buffer:
            flow.append(
                ListFlowable(
                    [ListItem(Paragraph(item, style_p)) for item in bullet_buffer],
                    bulletType="bullet",
                    leftIndent=16,
                    bulletFontName="Helvetica",
                    bulletFontSize=9,
                    bulletOffsetY=2,
                )
            )
            flow.append(Spacer(1, 6))
            bullet_buffer = []

    def flush_numbers() -> None:
        nonlocal number_buffer
        if number_buffer:
            flow.append(
                ListFlowable(
                    [ListItem(Paragraph(item, style_p)) for item in number_buffer],
                    bulletType="1",
                    start="1",
                    leftIndent=16,
                    bulletFontName="Helvetica",
                    bulletFontSize=9,
                    bulletOffsetY=2,
                )
            )
            flow.append(Spacer(1, 6))
            number_buffer = []

    for kind, value in parsed:
        if kind not in {"bullet"}:
            flush_bullets()
        if kind not in {"number"}:
            flush_numbers()

        if kind == "h1":
            flow.append(Paragraph(value, style_title))
        elif kind == "h2":
            flow.append(Paragraph(value, style_h2))
        elif kind == "h3":
            flow.append(Paragraph(value, style_h3))
        elif kind == "p":
            flow.append(Paragraph(value, style_p))
        elif kind == "blank":
            flow.append(Spacer(1, 4))
        elif kind == "bullet":
            bullet_buffer.append(value)
        elif kind == "number":
            number_buffer.append(value)

    flush_bullets()
    flush_numbers()

    doc.build(flow)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    markdown_path = project_root / "report.md"
    output_path = project_root / "report.pdf"

    if not markdown_path.exists():
        raise FileNotFoundError(f"Report markdown not found: {markdown_path}")

    build_pdf(markdown_path=markdown_path, output_path=output_path)
    print(f"Generated PDF: {output_path}")


if __name__ == "__main__":
    main()
