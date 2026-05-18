from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response  
import pandas as pd
import io
import logging
from pipeline import SmartAutoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Auto Data Cleaning API")

@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok", "message": "API is up and running."}

@app.post("/clean")
async def clean_data(file: UploadFile = File(...), target: str = Query(None, description="Optional target column")):
    """Uploads a CSV and instantly returns a cleaned downloadable CSV file."""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=True) 
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or formatted incorrectly.")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file contains no data.")

    
    if target:
        if target not in df.columns:
            normalized_columns = df.columns.str.lower().str.replace(r'\s+', '_', regex=True)
            normalized_target = target.lower().replace(' ', '_')
            if normalized_target not in normalized_columns:
                raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset")

    try:
        logger.info(f"Running Cleaning Pipeline for {file.filename}")
        
        
        pipeline = SmartAutoPipeline(df, target=target)
        pipeline.clean_columns()
        pipeline.analyze()
        pipeline.handle_missing()
        pipeline.encode()
        pipeline.handle_outliers()
        pipeline.scale()

        
        cleaned_df = pipeline.get_data()
        
        
        cleaned_csv_string = cleaned_df.to_csv(index=False)

        
        return Response(
            content=cleaned_csv_string,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cleaned_{file.filename}"'}
        )

    except Exception as e:
        logger.error(f"Pipeline failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during data processing: {str(e)}")