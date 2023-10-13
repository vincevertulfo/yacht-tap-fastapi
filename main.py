from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import numpy as np
import pickle
import json

MODEL_PATH = 'final_model.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))

app = FastAPI()

@app.get('/')
async def index():
    return "Hello from Yacht App"

@app.post('/predict')
async def predict_time_to_burn(request: Request):
    
    # Read the raw JSON data from the request body
    raw_json = await request.body()
    
    # Convert the raw JSON data to a Python dictionary
    try:
        json_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON"}
    
    wind_speed = json_data.get('wind_speed')                    # UNIT: nautical miles / hour
    time_remaining = json_data.get('time_remaining')            # UNIT: seconds
    true_wind_angle = json_data.get('true_wind_angle')          # UNIT: degree
    
    input = np.array([[true_wind_angle, wind_speed]])
    boat_speed = model.predict(input)[0]
    
    distance = json_data.get('distance')                        # UNIT: miles -> nautical miles

    time_to_burn = time_remaining - (distance / boat_speed)     # UNIT: seconds
    
    data =  {
            'boat_speed' : boat_speed,
            'distance' : distance,
            'time_remaining' : time_remaining,
            'time_to_burn' : time_to_burn
        }
    return JSONResponse(
        content=data
        )   
    