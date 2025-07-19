from fastapi import FastAPI,Path, HTTPException, Query
import json

app = FastAPI()

def load_data():
    with open('patients.json','r') as f:
        return json.load(f)


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