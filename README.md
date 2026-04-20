# ◈ AI Business Copilot

**Autonomous Data Intelligence & Strategic Analytics Dashboard**

The AI Business Copilot is a high-integrity, automated analytics platform designed to transform raw datasets into executive-level strategy reports. Using a combination of Deep Learning (PyTorch) and Large Language Models (Groq), it autonomously classifies data, builds predictive models, and generates actionable business insights in real-time.

---

## 🚀 Key Features

- **🎯 AI Expert Routing**: Automatically detects your data domain (Finance, HR, or General) and assigns a specialized AI "Persona" (CFO, CHRO, or Strategic Director) to lead the analysis.
- **🧠 Diagnostic Intelligence**: Trains a custom PyTorch Neural Network on your dataset to identify key performance drivers using SHAP values and permutation importance.
- **📊 Premium Visualizations**: Generates 9+ mission-critical diagnostic charts, from distribution histograms to ML residual plots, featuring a sleek "Cyber-Aura" theme.
- **📝 Executive Strategy**: Not just data—strategy. The copilot generates professional Markdown reports with causal inference, explaining the "Why" behind your trends.
- **📜 Professional PDF Reporting**: One-click generation of premium, high-integrity PDF reports for executive presentation.
- **💬 Conversational BI**: A built-in "Data Chatter" that uses exact Pandas-computed math to answer your questions accurately without LLM "math hallucinations."

---

## 🛠 Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, Scikit-learn, Pandas, NumPy, SHAP
- **Artificial Intelligence**: Groq API (Llama-3.1, Mixtral)
- **Reporting**: FPDF2, Matplotlib, Seaborn
- **Frontend**: Vanilla JavaScript, CSS3 (Premium Dark-Glass Aesthetics)
- **Deployment**: Render, Docker, Gunicorn

---

## 🏁 Quick Start

### 1. Prerequisites
- Python 3.10+
- A Groq API Key (Get it from [console.groq.com](https://console.groq.com/))

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/A-Hooda/AI-Business-Copilot.git
cd AI-Business-Copilot/Data
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the `Data/` directory:
```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run Locally
```bash
python server.py
```
Open your browser to `http://localhost:8000` to access the dashboard.

---

## 🐳 Docker Deployment
The project is optimized for containerized environments. To run using Docker Compose:
```bash
cd Data
docker-compose up --build
```

---

## ☁️ Deployment on Render
This project is pre-configured for deployment on **Render**:
1. Connect your GitHub repository to Render.
2. Select **Docker** as the environment.
3. Add your `GROQ_API_KEY` to the **Environment Variables**.
4. The system will automatically use the dynamic `$PORT` and configure Gunicorn for production scalability.

---

## 🛡 Stability Features (Turbo Mode)
- **Defensive Indexing**: Robust column validation prevents crashes even with complex headers or AI misnaming.
- **Memory Management**: Automatic garbage collection and plot cleanup to handle large files on memory-constrained cloud instances.
- **Timeout Resilience**: Configured for extended processing windows (300s) to ensure complex analysis completes successfully.

---

*Developed with focus on High-Integrity Analytics and Strategic Decision Support.*
