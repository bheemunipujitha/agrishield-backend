from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import UUID
from dotenv import load_dotenv
import httpx
import numpy as np
import io
import os

load_dotenv()

app = FastAPI(title="AGRISHIELD AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model", "plant_disease_efficientnet.keras"
)
_model = None
_model_load_error = None


def _get_disease_model():
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Disease image analysis is unavailable because TensorFlow could not "
                "load in this environment. Marketplace and Market Report endpoints "
                "can still run."
            ),
        ) from _model_load_error
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        return _model
    except Exception as exc:
        _model_load_error = exc
        raise HTTPException(
            status_code=503,
            detail=(
                "Disease image analysis is unavailable because TensorFlow could not "
                "load in this environment. Marketplace and Market Report endpoints "
                "can still run."
            ),
        ) from exc


class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy",
]


@app.get("/")
def home():
    return {"message": "AGRISHIELD AI Backend is running!"}


@app.post("/voice-analyze")
async def voice_analyze(data: dict):
    text = data.get("text", "")
    return {
        "message": "Voice received successfully",
        "voice_text": text,
        "recommendation": "Please upload a crop image for AI disease detection.",
    }


@app.post("/analyze")
async def analyze_crop(file: UploadFile = File(...)):
    image_data = await file.read()
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    model = _get_disease_model()
    prediction = model.predict(image_array, verbose=0)
    predicted_index = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_index] * 100)
    return {
        "disease": class_names[predicted_index],
        "confidence": round(confidence, 2),
        "recommendation": "Monitor affected leaves and verify the diagnosis with an agricultural expert.",
    }


# -----------------------------------------------------------------------------
# Shared Marketplace API (Supabase Postgres + Row Level Security)
# -----------------------------------------------------------------------------

def _marketplace_configuration():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="Marketplace database is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY in backend environment variables.",
        )
    return url, key


class MarketplaceListingCreate(BaseModel):
    crop_name: str = Field(min_length=1, max_length=100)
    variety: Optional[str] = Field(default=None, max_length=100)
    quantity: float = Field(gt=0, le=100000000)
    unit: Literal["kg", "quintal", "tonne", "bag", "crate"] = "kg"
    expected_price: float = Field(ge=0, le=1000000000)
    quality: str = Field(default="Standard", min_length=1, max_length=80)
    location: str = Field(min_length=1, max_length=180)
    harvest_date: Optional[str] = None
    description: Optional[str] = Field(default=None, max_length=2000)


class MarketplaceListingUpdate(BaseModel):
    crop_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    variety: Optional[str] = Field(default=None, max_length=100)
    quantity: Optional[float] = Field(default=None, gt=0, le=100000000)
    unit: Optional[Literal["kg", "quintal", "tonne", "bag", "crate"]] = None
    expected_price: Optional[float] = Field(default=None, ge=0, le=1000000000)
    quality: Optional[str] = Field(default=None, min_length=1, max_length=80)
    location: Optional[str] = Field(default=None, min_length=1, max_length=180)
    harvest_date: Optional[str] = None
    description: Optional[str] = Field(default=None, max_length=2000)
    status: Optional[Literal["Available", "Negotiating", "Sold"]] = None


def _read_bearer_token(authorization: Optional[str], required: bool = False):
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="Sign in first to manage marketplace listings.")
        return None
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(status_code=401, detail="Authorization must use a Supabase Bearer access token.")
    return token.strip()


async def _supabase_rest_request(
    method: str, table_path: str, *, access_token: Optional[str] = None,
    params: Optional[dict] = None, body: Optional[dict] = None,
    prefer: Optional[str] = None,
):
    supabase_url, publishable_key = _marketplace_configuration()
    headers = {"apikey": publishable_key, "Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    if prefer:
        headers["Prefer"] = prefer
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.request(
                method, f"{supabase_url}/rest/v1/{table_path.lstrip('/')}",
                headers=headers, params=params, json=body,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the marketplace database.") from exc
    if response.status_code >= 400:
        try:
            upstream_detail = response.json()
        except ValueError:
            upstream_detail = response.text[:500]
        status = response.status_code if response.status_code < 500 else 502
        raise HTTPException(
            status_code=status,
            detail={"message": "Marketplace database request failed.", "database_response": upstream_detail},
        )
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _safe_ilike(value: str) -> str:
    cleaned = value.strip()
    for special in ("*", "%", ",", "(", ")"):
        cleaned = cleaned.replace(special, " ")
    return f"ilike.*{cleaned.strip()}*"


@app.get("/marketplace/listings")
async def get_marketplace_listings(
    crop: Optional[str] = Query(default=None, max_length=100),
    location: Optional[str] = Query(default=None, max_length=180),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    authorization: Optional[str] = Header(default=None),
):
    token = _read_bearer_token(authorization, required=False)
    params = {
        "select": "id,crop_name,variety,quantity,unit,expected_price,quality,location,harvest_date,description,status,created_at,updated_at",
        "status": "eq.Available", "order": "created_at.desc",
        "limit": str(limit), "offset": str(offset),
    }
    if crop:
        params["crop_name"] = _safe_ilike(crop)
    if location:
        params["location"] = _safe_ilike(location)
    return await _supabase_rest_request("GET", "crop_listings", access_token=token, params=params)


@app.get("/marketplace/my-listings")
async def get_my_marketplace_listings(
    owner_id: UUID = Query(..., description="Current signed-in user's Supabase user ID"),
    authorization: Optional[str] = Header(default=None),
):
    token = _read_bearer_token(authorization, required=True)
    params = {
        "select": "id,owner_id,crop_name,variety,quantity,unit,expected_price,quality,location,harvest_date,description,status,created_at,updated_at",
        "owner_id": f"eq.{owner_id}", "order": "created_at.desc",
    }
    return await _supabase_rest_request("GET", "crop_listings", access_token=token, params=params)


@app.post("/marketplace/listings", status_code=201)
async def create_marketplace_listing(
    listing: MarketplaceListingCreate,
    authorization: Optional[str] = Header(default=None),
):
    token = _read_bearer_token(authorization, required=True)
    body = jsonable_encoder(listing.dict(exclude_none=True))
    body["status"] = "Available"
    result = await _supabase_rest_request(
        "POST", "crop_listings", access_token=token, body=body,
        prefer="return=representation",
    )
    if not result:
        raise HTTPException(status_code=502, detail="Listing was not returned after saving.")
    return result[0] if isinstance(result, list) else result


@app.patch("/marketplace/listings/{listing_id}")
async def update_marketplace_listing(
    listing_id: UUID, listing: MarketplaceListingUpdate,
    authorization: Optional[str] = Header(default=None),
):
    token = _read_bearer_token(authorization, required=True)
    body = jsonable_encoder(listing.dict(exclude_unset=True, exclude_none=True))
    if not body:
        raise HTTPException(status_code=400, detail="Provide at least one field to update.")
    from datetime import datetime, timezone
    body["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = await _supabase_rest_request(
        "PATCH", "crop_listings", access_token=token,
        params={"id": f"eq.{listing_id}", "select": "*"}, body=body,
        prefer="return=representation",
    )
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found or you do not own it.")
    return result[0] if isinstance(result, list) else result


@app.delete("/marketplace/listings/{listing_id}")
async def delete_marketplace_listing(
    listing_id: UUID, authorization: Optional[str] = Header(default=None),
):
    token = _read_bearer_token(authorization, required=True)
    result = await _supabase_rest_request(
        "DELETE", "crop_listings", access_token=token,
        params={"id": f"eq.{listing_id}", "select": "id"},
        prefer="return=representation",
    )
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found or you do not own it.")
    return {"message": "Listing deleted successfully.", "deleted": result[0] if isinstance(result, list) else result}


# -----------------------------------------------------------------------------
# Market Report API (Government of India data.gov.in / AGMARKNET dataset)
# Set DATA_GOV_IN_API_KEY in backend .env / deployment environment.
# This dataset is updated by reporting markets; it is daily data, not tick-by-tick live.
# -----------------------------------------------------------------------------
DATA_GOV_IN_RESOURCE_URL = (
    "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
)


@app.get("/market-report/prices")
async def get_market_report_prices(
    state: Optional[str] = Query(default=None, max_length=100),
    district: Optional[str] = Query(default=None, max_length=100),
    market: Optional[str] = Query(default=None, max_length=100),
    commodity: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Fetch reported mandi prices from the official data.gov.in resource."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Market Report is not configured yet. Add DATA_GOV_IN_API_KEY to the backend .env file.",
        )

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "offset": offset,
    }
    # data.gov.in uses filters[field] for resource-field filtering.
    for field, value in (
        ("state", state), ("district", district),
        ("market", market), ("commodity", commodity),
    ):
        if value and value.strip():
            params[f"filters[{field}]"] = value.strip()

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.get(DATA_GOV_IN_RESOURCE_URL, params=params)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not reach data.gov.in right now. Please try again later.",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                "data.gov.in rejected the Market Report request. Check the API key "
                "and dataset availability in your data.gov.in account."
            ),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="data.gov.in returned an unreadable response.") from exc

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise HTTPException(status_code=502, detail="data.gov.in returned an unexpected records format.")

    return {
        "source": "Government of India data.gov.in / AGMARKNET",
        "resource_id": "9ef84268-d588-465a-a308-a864a43d0070",
        "data_type": "reported daily mandi prices; availability may vary by market",
        "count": len(records),
        "total": payload.get("total"),
        "limit": limit,
        "offset": offset,
        "filters": {
            "state": state, "district": district,
            "market": market, "commodity": commodity,
        },
        "records": records,
    }
