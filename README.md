# AI Business Copilot

**Autonomous Data Intelligence and Strategic Analytics Platform**

The AI Business Copilot is a sophisticated analytics solution designed to bridge the gap between complex data processing and executive decision-making. By integrating advanced machine learning with large language models, the platform transforms raw datasets into professional, strategy-oriented insights.

## Project Overview

In today's data-driven environment, the challenge is often not a lack of data, but the difficulty in extracting meaningful, high-level narratives from it. This platform automates the entire analytical lifecycle—from domain identification and data cleaning to predictive modeling and strategic reporting—ensuring that business leaders receive accurate, actionable information in a clear and professional format.

## The Analytical Lifecycle: How It Works

To ensure both technical depth and user accessibility, the platform follows a structured four-stage analytical pipeline.

### 1. Contextual Ingestion and Interpretation
When a file is uploaded, the system does not just read the numbers; it uses an artificial intelligence "Interpreter" to understand the functional role of every column. By analyzing headers and sample data, it identifies which columns represent your primary goals (Metrics), your categories (Dimensions), and your timeline (Temporal Axis). This allows the system to pivot its logic dynamically based on whether you are analyzing financial ledgers or HR payrolls.

### 2. Defensive Data Engineering
Data is rarely perfect. The platform includes an automated "Adapter" that performs high-integrity cleaning. It handles missing values using statistical medians, standardizes date formats, and resolves currency symbols. Most importantly, it uses "Defensive Indexing" to ensure that even if a column name is slightly inconsistent, the system can still find and process the correct data without crashing.

### 3. Neural Diagnostic Modeling (The "Why" Engine)
At the heart of the platform is a custom PyTorch Neural Network. Unlike a simple calculator, this model "learns" the complex relationships between all your business factors. 
- **Predictive Power**: The model attempts to predict your primary metric (like Revenue) based on all other variables. 
- **Driver Identification**: We use a technique called SHAP (SHapley Additive exPlanations). Think of this as a "What-If" machine that mathematically determines exactly how much "credit" each factor deserves for your success. For example, it can quantify exactly how much a specific region or department contributed to a boost in sales.

### 4. Causal Reasoning and Strategic Narration
Once the numbers are computed, the Strategic Advisor (LLM) takes over. Instead of just listing facts, it performs "Causal Inference." It looks at the trends and uses its deep business knowledge to suggest *why* they happened—considering factors like seasonal demand, market shifts, or operational efficiency. This results in a narrative report that sounds like it was written by a senior consultant.

## Technical Architecture

- **Backend Framework**: Python 3.11 with FastAPI for high-performance, asynchronous service delivery.
- **Machine Learning Layer**: PyTorch for diagnostic modeling and Scikit-learn for rapid feature analysis.
- **Artificial Intelligence**: Integration with Groq-hosted Llama and Mixtral models for sophisticated reasoning.
- **Computation Engine**: A dedicated system that performs exact mathematical calculations via Pandas *before* the AI interprets them, eliminating the risk of mathematical errors common in standard AI models.
- **Visualization Suite**: Custom-themed Matplotlib and Seaborn implementations for professional-grade analytics.

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- A valid Groq API Key (Available via the Groq Cloud Console)

### Local Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/A-Hooda/AI-Business-Copilot.git
   ```
2. Navigate to the source directory and install dependencies:
   ```bash
   cd AI-Business-Copilot/Data
   pip install -r requirements.txt
   ```
3. Configure your environment by creating a `.env` file:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. Launch the application:
   ```bash
   python server.py
   ```

## Deployment

### Docker Configuration
The repository includes a Dockerfile and docker-compose.yml optimized for containerized environments:
```bash
docker-compose up --build
```

### Cloud Deployment (Render)
The platform is pre-configured for deployment on Render:
1. Connect this repository to your Render account.
2. Select Docker as the runtime.
3. Configure the `GROQ_API_KEY` in your environment variables.
The platform will automatically handle dynamic port binding and production-grade scaling via Gunicorn.

---
*Developed with a focus on data integrity, strategic depth, and professional-grade business intelligence.*
