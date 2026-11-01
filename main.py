import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Folder this file lives in, so paths work no matter where uvicorn is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Load the trained pipeline once, when the server starts (cleaning + model together)
model = joblib.load(os.path.join(BASE_DIR, "phone_model.joblib"))


class Phone(BaseModel):
    """The JSON the web page sends. Optional means the field may be left empty."""
    battery: Optional[float] = None
    storage: Optional[float] = None
    brand: Optional[str] = None


def or_nan(value):
    """Empty fields become NaN, which the pipeline's imputers know how to fill."""
    return np.nan if value is None else value


@app.post("/predict")
def predict(phone: Phone):
    # Column names must match the training data exactly: Battery, Storage, Brand
    row = pd.DataFrame({
        "Battery": [or_nan(phone.battery)],
        "Storage": [or_nan(phone.storage)],
        "Brand": [or_nan(phone.brand)],
    })
    try:
        price = model.predict(row)[0]
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
    return {"price": round(float(price))}


# Keep this LAST: it serves static/index.html at "/" and would otherwise catch /predict
app.mount(
    "/",
    StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True),
    name="static",
)                     # no extra fit needed, the grid already retrained the winner