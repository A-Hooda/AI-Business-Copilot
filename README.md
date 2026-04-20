# AI Business Copilot
### *Autonomous Data Intelligence and Strategic Analytics Platform*

---

> [!NOTE]
> The AI Business Copilot is a sophisticated analytics solution designed to bridge the gap between complex data processing and executive decision-making. By integrating advanced machine learning with large language models, the platform transforms raw datasets into professional, strategy-oriented insights.

---

## Project Overview

In today's data-driven environment, the challenge is often not a lack of data, but the difficulty in extracting meaningful, high-level narratives from it. This platform automates the entire analytical lifecycle—from domain identification and data cleaning to predictive modeling and strategic reporting—ensuring that business leaders receive accurate, actionable information in a clear and professional format.

---

## The Analytical Lifecycle: How It Works

To ensure both technical depth and user accessibility, the platform follows a structured four-stage analytical pipeline:

1. **Contextual Ingestion and Interpretation**
   - When a file is uploaded, an artificial intelligence "Interpreter" understands the functional role of every column.
   - It identifies **Metrics** (goals), **Dimensions** (categories), and **Temporal Axes** (timelines).
   - This allows the system to pivot its logic dynamically based on whether you are analyzing financial ledgers or HR payrolls.

2. **Defensive Data Engineering**
   - The automated "Adapter" performs high-integrity cleaning, handling missing values using statistical medians and standardizing formats.
   - **Defensive Indexing**: Ensures that even if a column name is slightly inconsistent, the system can still find and process the correct data without crashing.

3. **Neural Diagnostic Modeling (The "Why" Engine)**
   - At the heart of the platform is a custom **PyTorch Neural Network** that "learns" the complex relationships between all your business factors.
   - **Predictive Power**: The model predicts your primary metric based on all other variables.
   - **Driver Identification (SHAP)**: A "What-If" machine that mathematically determines exactly how much "credit" each factor deserves for your success (e.g., quantifying a specific region's contribution to total sales).

4. **Causal Reasoning and Strategic Narration**
   - Once the numbers are computed, the Strategic Advisor (LLM) performs **Causal Inference**.
   - It suggests *why* trends happened by considering seasonal demand, market shifts, or operational efficiency.
   - The result is a narrative report that sounds like it was written by a senior consultant.

---

## Technical Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | Python 3.11 / FastAPI | High-performance, asynchronous service delivery. |
| **Machine Learning** | PyTorch / Scikit-learn | Diagnostic modeling and rapid feature analysis. |
| **Intelligence** | Groq (Llama / Mixtral) | Sophisticated reasoning and strategic narration. |
| **Analytics Engine** | Pandas / NumPy | Robust data pipelines with exact mathematical calculation. |
| **Visualizations** | Matplotlib / Seaborn | Custom-themed, professional-grade analytics suite. |

---

## Setup and Installation

### Prerequisites
* Python 3.10 or higher
* A valid Groq API Key (Available via the [Groq Cloud Console](https://console.groq.com/))

### Local Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/A-Hooda/AI-Business-Copilot.git
   ```
2. **Install dependencies:**
   ```bash
   cd AI-Business-Copilot/Data
   pip install -r requirements.txt
   ```
3. **Configure Environment:**
   Create a `.env` file in the `Data/` directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. **Launch Application:**
   ```bash
   python server.py
   ```

---

## Deployment

### Docker Configuration
The repository is optimized for containerized environments:
```bash
docker-compose up --build
```

### Cloud Deployment (Render)
The platform is pre-configured for seamless deployment on Render:
1. Connect this repository to your Render account.
2. Select **Docker** as the runtime.
3. Configure the `GROQ_API_KEY` in your environment variables.

*The platform automatically handles dynamic port binding and production-grade scaling via Gunicorn.*

---
> *Developed with a focus on data integrity, strategic depth, and professional-grade business intelligence.*
