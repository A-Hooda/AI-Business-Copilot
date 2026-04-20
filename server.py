import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import uuid
import gc
import matplotlib.pyplot as plt

# Project Imports
from ingester import GenericIngester
from router import ExpertRouter
from interpreter import DataInterpreter
from adapter import DataAdapter
from predictor import UniversalPredictor
from visualizer import AgnosticVisualizer
from advisor import ExpertAdvisor
from chat_engine import DataChatter
from pdf_generator import generate_professional_pdf

# Specialists
from experts.finance_expert import FinanceExpert
from experts.hr_expert import HRExpert
from experts.general_expert import GeneralExpert

app = FastAPI(title="Sourcedotcom AI Analyst", version="2.0.0")

# Global in-memory session store for scalability
sessions = {}

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    try:
        session_id = str(uuid.uuid4())
        session_dir = f"reports/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. Save uploaded file instantly
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Basic Ingestion to confirm it works
        df = GenericIngester.load_data(file_path)
        
        # Initialize session state
        sessions[session_id] = {
            "status": "processing",
            "progress": 10,
            "filename": file.filename,
            "df": df,
            "domain": "analyzing...",
            "title": "SCANNING...",
            "results": None
        }

        # 3. Queue the heavy lifting
        background_tasks.add_task(run_automated_analysis, session_id, file_path)

        return {"session_id": session_id, "status": "processing"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def run_automated_analysis(session_id: str, file_path: str):
    """
    Heavy-duty background task for deep data intelligence.
    """
    try:
        session_data = sessions[session_id]
        df = session_data["df"]
        session_dir = f"reports/{session_id}"

        # Capture RAW data health BEFORE any cleaning/imputation
        total_nas = df.isna().sum().sum()
        raw_missing_pct = f"{(total_nas / df.size * 100):.2f}%" if df.size > 0 else "0.00%"
        
        # 1. Domain Routing
        session_data["progress"] = 20
        router = ExpertRouter()
        domain = router.determine_domain(df)
        session_data["domain"] = domain

        if domain == 'finance':
            session_data["expert"] = FinanceExpert()
            session_data["title"] = "FINANCIAL ANALYST"
        elif domain == 'hr':
            session_data["expert"] = HRExpert()
            session_data["title"] = "HR ANALYST"
        else:
            session_data["expert"] = GeneralExpert()
            session_data["title"] = "GENERAL ANALYST"

        # 2. Intelligence Mapping
        session_data["progress"] = 40
        interpreter = DataInterpreter()
        mapping = interpreter.identify_roles(df)
        roles = mapping.get("column_roles", {})
        session_data["roles"] = roles
        session_data["currency_symbol"] = mapping.get("currency_symbol", "$")
        
        # 3. Clean and Prep
        df, adapter_currency = DataAdapter.auto_clean(df, roles)
        if adapter_currency and (not session_data["currency_symbol"] or session_data["currency_symbol"] == "$"):
            session_data["currency_symbol"] = adapter_currency
            
        df = session_data["expert"].preprocess(df, roles)
        session_data["df"] = df

        # 4. ML Drivers (The Heavy Part)
        session_data["progress"] = 60
        df_scaled = DataAdapter.scale_for_ml(df, roles)
        prediction_payload = UniversalPredictor.get_performance_drivers(df_scaled, roles)
        if prediction_payload:
            drivers, y_test, y_pred, mae, r2 = prediction_payload
        else:
            drivers, y_test, y_pred, mae, r2 = {}, None, None, 0.0, 0.0
            
        forecast_df = UniversalPredictor.generate_forecast(df, roles)
        
        # 5. Visualizer
        session_data["progress"] = 80
        AgnosticVisualizer.create_reports(df, roles, drivers, y_test, y_pred, forecast_df=forecast_df, output_dir=session_dir)
        
        # Explicitly free memory after heavy chart generation
        plt.close('all')
        if 'df_scaled' in locals(): del df_scaled
        gc.collect()

        # 6. Strategy & Summary
        advisor = ExpertAdvisor()
        persona = session_data["expert"].get_consultant_prompt()
        final_analysis = interpreter.identify_roles(df, custom_persona=persona)
        insight = final_analysis.get("business_brief", "No insight generated.")
        strategy = advisor.generate_strategy(df, roles, drivers, domain, insight, session_data.get("currency_symbol", "$"))

        target_col = next((c for c, r in roles.items() if r == 'primary_metric' and c in df.columns), None)
        dim_col = next((c for c, r in roles.items() if r == 'primary_dimension' and c in df.columns), None)
        
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_formatted = f"{(mem_bytes / 1024**2):.2f} MB" if mem_bytes > 1024*1024 else f"{(mem_bytes / 1024):.2f} KB"

        summary = {
            "records": f"{len(df):,}",
            "columns": len(df.columns),
            "numeric_feats": len(df.select_dtypes(include=['number']).columns),
            "categorical_feats": len(df.select_dtypes(exclude=['number']).columns),
            "duplicates": f"{int(df.duplicated().sum()):,}",
            "missing_pct": raw_missing_pct,
            "memory": mem_formatted,
            "top_metric": "—",
            "top_segment": "—",
            "mae": f"{mae:.3f}" if mae else "N/A",
            "r2": f"{r2:.3f}" if r2 else "N/A"
        }
        
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            summary["top_metric"] = f"{df[target_col].mean():.2f}"
            if dim_col and dim_col in df.columns:
                try:
                    summary["top_segment"] = str(df.groupby(dim_col)[target_col].sum().idxmax())
                except Exception: pass

        result = {
            "session_id": session_id,
            "title": session_data["title"],
            "insight": insight,
            "strategy": strategy,
            "currency_symbol": session_data["currency_symbol"],
            "summary": summary
        }
        
        session_data["last_result"] = result
        session_data["status"] = "completed"
        session_data["progress"] = 100
        
        # Final cleanup for this session's background memory
        gc.collect()
        
    except Exception as e:
        traceback.print_exc()
        if session_id in sessions:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = str(e)


@app.get("/analysis-status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "status": session["status"],
        "progress": session.get("progress", 0),
        "result": session.get("last_result"),
        "error": session.get("error")
    }




class ChatRequest(BaseModel):
    session_id: str = None
    question: str = None

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"--- [Chat] Request: {request.model_dump()} ---")
    session_id = request.session_id
    question = request.question
    
    if not session_id or not question:
        return {"answer": "⚠️ Error: Missing session info or question."}
    
    session = sessions.get(session_id)
    if not session or session.get("df") is None:
        return {"answer": "⚠️ Invalid or expired session. Please upload your dataset again!"}

    try:
        chatter = DataChatter()
        answer = chatter.ask(session["df"], session["roles"], question, session.get("currency_symbol", "$"))
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"❌ Logic Error: {str(e)}"}


@app.get("/download-pdf/{session_id}")
async def download_pdf(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    session_data = sessions[session_id]
    result = session_data.get("last_result")
    if not result:
        return JSONResponse({"error": "Analysis not complete"}, status_code=400)
    
    session_dir = os.path.join("reports", session_id)
    pdf_path = os.path.join(session_dir, "Executive_Insight_Report.pdf")
    
    # Generate on the fly
    generate_professional_pdf(pdf_path, result, session_dir)
    
    return FileResponse(
        pdf_path, 
        media_type='application/pdf', 
        filename=f"Executive_Report_{session_id[:8]}.pdf"
    )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
