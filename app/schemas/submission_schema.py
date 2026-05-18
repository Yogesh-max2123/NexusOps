from pydantic import BaseModel, Field
from datetime import datetime

class SubmissionResponse(BaseModel):
    id: str = Field(alias="_id")  
    user_id: str
    dataset_url: str
    target_column: str
    use_case: str
    requirement: str
    status: str
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }
