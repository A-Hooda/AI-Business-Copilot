# AI Business Copilot

**Autonomous Data Intelligence and Strategic Analytics Platform**

The AI Business Copilot is a sophisticated analytics solution designed to bridge the gap between complex data processing and executive decision-making. By integrating advanced machine learning with large language models, the platform transforms raw datasets into professional, strategy-oriented insights.

## Project Overview

In today's data-driven environment, the challenge is often not a lack of data, but the difficulty in extracting meaningful, high-level narratives from it. This platform automates the entire analytical lifecycle—from domain identification and data cleaning to predictive modeling and strategic reporting—ensuring that business leaders receive accurate, actionable information in a clear and professional format.

## Core Capabilities

### Specialized Analytical Routing
The platform automatically identifies the domain of your dataset, whether it pertains to Finance, Human Resources, or General Business operations. It then assumes a specialized analytical persona—such as a CFO or CHRO—to ensure that the insights generated are industry-relevant and focused on the metrics that matter most.

### Predictive Modeling and Diagnostic Intelligence
Using PyTorch-based neural networks, the system identifies the key drivers behind your business performance. By utilizing SHAP values and permutation importance, the platform explains not just what is happening in your data, but why it is happening, providing a level of transparency often missing in standard automated tools.

### Executive Level Reporting
The final output is a comprehensive strategic brief. Unlike simple data summaries, these reports focus on causal inference and macro-economic factors, providing the kind of high-level reasoning expected in boardrooms. Reports can be exported as professional PDF documents for external distribution.

### Conversational Business Intelligence
An integrated chat interface allows for direct interrogation of the data. To resolve the common issue of mathematical inaccuracies in language models, this system utilizes a dedicated computation engine that performs exact calculations via Pandas before the AI interprets the results.

## Technical Architecture

- **Backend Framework**: Python 3.11 with FastAPI for high-performance, asynchronous service delivery.
- **Machine Learning Layer**: PyTorch for diagnostic modeling and Scikit-learn for rapid feature analysis.
- **Artificial Intelligence**: Integration with Groq-hosted Llama and Mixtral models for sophisticated reasoning.
- **Data Engineering**: Robust Pandas and NumPy pipelines for automated cleaning and transformation.
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
