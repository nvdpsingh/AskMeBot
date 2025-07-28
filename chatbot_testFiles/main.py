from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AskMeBot API", version="1.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
mongodb_client: AsyncIOMotorClient = None
database = None

@app.on_event("startup")
async def startup_event():
    """Connect to MongoDB on startup"""
    global mongodb_client, database
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "askmebot")
    
    mongodb_client = AsyncIOMotorClient(mongodb_url)
    database = mongodb_client[database_name]
    
    # Test connection
    try:
        await mongodb_client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MongoDB on shutdown"""
    if mongodb_client:
        mongodb_client.close()
        print("👋 Disconnected from MongoDB")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "AskMeBot API is running with MongoDB!",
        "database": "MongoDB"
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint with MongoDB integration"""
    if database:
        try:
            # Test database operation
            collections = await database.list_collection_names()
            return {
                "message": "MongoDB integration working!",
                "collections": collections,
                "database_name": database.name
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Database not connected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 