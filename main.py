# main.py
# ============================================================
# Tourism Recommendation API
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from contextlib import asynccontextmanager
import uvicorn

from model import load_data, load_config, run_pipeline

# ============================================================
# Startup: load data once into memory
# ============================================================
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["df"]     = load_data()
    app_state["config"] = load_config()
    print(f"✅ Data loaded: {len(app_state['df'])} places")
    yield
    app_state.clear()

# ============================================================
# App
# ============================================================
app = FastAPI(
    title       = "🗺️ Egypt Tourism Recommendation API",
    description = """
## Recommendation & Constraint Optimization System

يقوم النظام بتوصية خطة رحلة سياحية ذكية داخل مصر بناءً على:
- **الموقع** — المحافظة المستهدفة
- **الميزانية** — بالجنيه المصري
- **المدة** — عدد الأيام
- **الاهتمامات** — التصنيفات المفضلة

### الخوارزمية:
1. **Scoring** — ترتيب الأماكن بناءً على الاهتمامات والميزانية والشعبية
2. **K-Means Clustering** — توزيع جغرافي على الأيام
3. **Constraint Optimization** — موازنة الوقت والميزانية
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Schemas
# ============================================================
class TripRequest(BaseModel):
    governorate   : Literal["Cairo", "Giza", "Luxor", "Aswan"] = Field(
        ..., example="Luxor",
        description="المحافظة المستهدفة"
    )
    budget_egp    : float = Field(
        ..., gt=0, le=50000, example=3000,
        description="إجمالي الميزانية بالجنيه المصري"
    )
    duration_days : int = Field(
        ..., ge=1, le=14, example=3,
        description="عدد أيام الرحلة"
    )
    interests     : list[str] = Field(
        ..., min_length=1, example=["historical", "pharaonic", "temple", "museum"],
        description="قائمة الاهتمامات بالإنجليزية"
    )

    @field_validator("interests")
    @classmethod
    def interests_not_empty(cls, v):
        if not v or all(i.strip() == "" for i in v):
            raise ValueError("interests must contain at least one non-empty value")
        return [i.strip().lower() for i in v]

class HealthResponse(BaseModel):
    status        : str
    total_places  : int
    governorates  : list[str]
    version       : str

# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
def health_check():
    """التحقق من حالة السيرفر وعدد الأماكن المحملة."""
    df = app_state.get("df")
    return {
        "status"       : "ok",
        "total_places" : len(df) if df is not None else 0,
        "governorates" : ["Cairo", "Giza", "Luxor", "Aswan"],
        "version"      : "1.0.0",
    }

@app.post(
    "/recommend",
    summary="Generate Trip Itinerary",
    tags=["Recommendation"],
    response_description="خطة رحلة مقسمة لأيام مع التكاليف والإحداثيات",
)
def recommend(request: TripRequest):
    """
    ## توليد خطة رحلة سياحية مخصصة

    يستقبل بيانات المستخدم ويرجع:
    - **trip_summary**: ملخص الرحلة (التكلفة، الساعات، عدد الأماكن)
    - **itinerary**: الخطة مقسمة أيام (كل يوم: الأماكن + الإحداثيات + التكاليف)
    - **validation**: تقرير التحقق من جودة الخطة
    """
    try:
        result = run_pipeline(
            governorate   = request.governorate,
            budget_egp    = request.budget_egp,
            duration_days = request.duration_days,
            interests     = request.interests,
            df            = app_state["df"],
            config        = app_state["config"],
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# ============================================================
# Run locally
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
