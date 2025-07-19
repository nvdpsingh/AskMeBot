from fastapi import FastAPI,Path, HTTPException, Query
from
from fastapi.responses import JSONResponse
from typing import Annotated, Literal
from pydantic import BaseModel, Field, computed_field
import json

app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str, Field(...,description="Id of the patient", examples= ['P001'])]
    name: Annotated[str, Field(..., description="Name of the patient")]
    city: Annotated[str, Field(...,description="City where the patient is living")]
    age:Annotated[int,Field(..., gt=0,lt=120, description="Age of the patient")]
    gender: Annotated[Literal['male','female','others'], Field(..., description="Gender of the patient")]
    height:Annotated[int, Field(...,gt=0,description="Height of the patient in meters")]
    weight: Annotated[int, Field(...,gt=0,description="Weight of the patient in KGs")]

    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round(self.weight/(self.height**2),2)
        return bmi
    
    @computed_field
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return "Underweight"
        elif self.bmi < 25:
            return "Normal"
        elif self.bmi < 30:
            return "Normal"
        else:
            return "Obese"

def load_data():
    with open('patients.json','r') as f:
        return json.load(f)

def save_data(data):
    with open('patients.json','w') as f:
        json.dump(data,f)


@app.get("/")
def hello():
    return {'message':'Patient Management System API'}

@app.get('/about')
def about():
    return {'message':'A fully functional API to manage your pateint records'}

@app.get("/view")
def view():
    data = load_data()
    return data

@app.get("/patient/{patient_id}")
def pateint(patient_id:str = Path(...,description="Enter Patient Id here", example="P001")):
    #load all the pateints
    data = load_data()

    if patient_id.capitalize() in data:
        return data[patient_id.capitalize()]
    raise HTTPException(status_code=404,detail="Patient ID not found")



@app.get("/sort")
def sort_patients(sort_by:str = Query(...,description="sort on the basis of height, weight or bmi"), order:str = Query('asc', description="sort in asc or desc")):
    
    valid_field = ['height','weight','bmi']

    if sort_by not in valid_field:
        raise HTTPException(status_code=400, detail=f"Invalid Field, Select from {valid_field}")
    
    if order not in['asc','desc']:
        raise HTTPException(status_code=400,detail="Invalid Order, select from asc or desc")
    
    data = load_data()

    sort_order = True if order =="desc" else False

    sorted_data= sorted(data.values(), key = lambda x: x.get(sort_by,0),reverse=sort_order)

    return sorted_data

@app.post("/create")
def create_patient(patient: Patient):
    #load existing data

    data = load_data()

    #check if the patient
    if patient.id in data:
        raise HTTPException(status_code=400, detail="Patient already exists")
    
    #new patient add to the database
    data[patient.id] = patient.model_dump(exclude=['id']) 
 
    #save into json file
    save_data(data)


