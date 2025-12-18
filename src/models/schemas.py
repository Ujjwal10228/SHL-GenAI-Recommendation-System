# src/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    message: str = Field(default="API is running", example="API is running")

class AssessmentRecommendation(BaseModel):
    assessment_name: str = Field(..., description="Name of the assessment")
    assessment_url: str = Field(..., description="URL to the assessment on SHL catalog")
    test_type: Optional[str] = Field(default=None, description="Test type: K (Knowledge), P (Personality), C (Cognitive), etc.")
    duration_minutes: Optional[int] = Field(default=None, description="Duration in minutes")
    category: Optional[str] = Field(default=None, description="Assessment category")

class RecommendRequest(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Natural language query or job description text",
        example="I am hiring for Java developers who can collaborate with business teams"
    )
    jd_url: Optional[str] = Field(
        default=None,
        description="URL containing job description (will be fetched and parsed)",
        example="https://example.com/job/java-dev"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Need Java developer with collaboration skills"
            }
        }

class RecommendResponse(BaseModel):
    results: List[AssessmentRecommendation] = Field(
        ...,
        description="List of recommended assessments (5-10 items)"
    )
    query_processed: Optional[str] = Field(
        default=None,
        description="Normalized query used for recommendation"
    )
    count: int = Field(..., description="Number of recommendations")
