from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {'message':'Hello World!'}

@app.get('/about')
def about():
    return {'message':'I am Navdeep Singh and i am learning FastAPI'}