import os
from fpdf import FPDF
from datetime import datetime

class ExecutiveReportPDF(FPDF):
    def header(self):
        """
        The header is called automatically by FPDF on every page (manual or auto-break).
        We use it to draw the dark background BEFORE any other content is added.
        """
        # 1. Draw the Background Rectangle FIRST (Bottom Layer)
        self.set_fill_color(5, 8, 20)
        # Use full page dimensions
        self.rect(0, 0, 210, 297, "F")
        
        # 2. Draw the Sidebar Indicator
        self.set_fill_color(0, 255, 136) # Removed alpha argument causing crash
        self.rect(0, 0, 5, 297, "F")

        # 3. Add Header Text (Top Layer)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(167, 181, 201)
        self.set_y(5)
        self.set_x(10)
        self.cell(0, 10, "AI BUSINESS INSIGHTS", 0, 0, "L")
        self.set_x(-40)
        self.cell(0, 10, datetime.now().strftime("%Y-%m-%d"), 0, 1, "R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(167, 181, 201)
        self.cell(0, 10, f"Page {self.page_no()} | Confidential Intellectual Property of Hooda.com", 0, 0, "C")

    def chapter_title(self, label, size=16):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(0, 255, 136) # Mint Accent
        self.cell(10) # Sidebar offset
        self.multi_cell(0, 10, label.upper())
        self.ln(5)

    def chapter_body(self, body, is_sub=False):
        self.set_font("Helvetica", "B" if is_sub else "", 12 if is_sub else 10)
        self.set_text_color(242, 247, 251) # White text
        if is_sub:
            self.set_text_color(0, 229, 255) # Subheading Cyan
        
        self.cell(10) # Sidebar offset
        self.multi_cell(0, 7, body)
        self.ln(2)

def generate_professional_pdf(output_path, data, session_dir):
    """
    Constructs a premium PDF report with embedded graphs and styled MD content.
    """
    pdf = ExecutiveReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Adding pages will now trigger the background draw in the header() automatically
    pdf.add_page()

    # Title
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(0, 255, 136)
    pdf.cell(10)
    pdf.cell(0, 20, "EXECUTIVE STRATEGY REPORT", 0, 1, "L")
    pdf.ln(10)

    # 1. Section: Data Overview
    pdf.chapter_title("SECTION 1: Data Overview")
    if data.get("summary"):
        s = data["summary"]
        summary_text = (
            f"Total Records: {s['records']} | Total Columns: {s['columns']}\n"
            f"Numerical Features: {s['numeric_feats']} | Categorical Features: {s['categorical_feats']}\n"
            f"Missing Ratio: {s['missing_pct']} | Memory: {s['memory']}\n"
            f"Model Evaluation -> MAE: {s.get('mae', 'N/A')} | R2 Score: {s.get('r2', 'N/A')}"
        )
        pdf.chapter_body(summary_text)

    # 2. Section: Insight & Strategy
    pdf.ln(10)
    pdf.chapter_title("SECTION 2: LLM Strategic Insights")
    
    pdf.chapter_body("PRIMARY BUSINESS INSIGHT:", is_sub=True)
    pdf.chapter_body(data.get("insight", "No insight provided."))
    
    pdf.ln(5)
    pdf.chapter_title("SECTION 3: Detailed Strategy")
    # Clean markdown artifacts like ** and ### for PDF compatibility
    strategy = data.get("strategy", "").replace("**", "").replace("###", "").replace("##", "")
    pdf.chapter_body(strategy)

    # 3. Visuals Section
    pdf.add_page()
    pdf.chapter_title("SECTION 4: Diagnostic Visualizations")
    
    # List of expected charts
    charts = [
        ("dist.png", "Metric Distribution Breakdown"),
        ("box.png", "Segment Variance Analysis"),
        ("correlation.png", "Feature Correlation Heatmap"),
        ("trend.png", "Historical Performance Trajectory"),
        ("forecast.png", "30-Day Predictive Horizon"),
        ("scatter.png", "Top Driver Dependency Map"),
        ("pred_vs_actual.png", "Neural Net Accuracy Validation"),
        ("residual.png", "Predictive Bias Analysis"),
        ("pie.png", "Categorical Volume Composition"),
        ("interaction.png", "SHAP / Feature Interaction weights")
    ]

    for filename, title in charts:
        img_path = os.path.join(session_dir, filename)
        if os.path.exists(img_path):
            if pdf.get_y() > 200: # New page if space is tight
                pdf.add_page()
                pdf.ln(10)

            pdf.chapter_body(title, is_sub=True)
            # Center image
            pdf.image(img_path, x=25, w=160)
            pdf.ln(10)

    pdf.output(output_path)
    return output_path
