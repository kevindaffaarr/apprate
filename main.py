from fastapi import FastAPI, Response
from numpy import full
import database as db
import json

"""
=============================
FAST API APP
Start the server by run this code in terminal
uvicorn main:app --reload
=============================
"""

# INITIATE APP
app = FastAPI(
	debug=True,
	title="apprate",
	version="0.0.0",
	contact={
		"name": "Kevin Daffa Arrahman",
		"email": "kevindaffaarr@quantist.io"
	}
)

@app.get("/")
async def home():
	return {"message": "Welcome to Apprate"}

@app.post("/apprate")
async def apprate(asset_appraised: db.AssetAppraised):
	full_appraisal = db.FullAppraisal(asset_appraised)
	response = {
		"asset_appraised_market_value" : full_appraisal.asset_appraised_market_value,
		"dp_full" : full_appraisal.top_market_data.to_dict(orient="index")
	}
	return response