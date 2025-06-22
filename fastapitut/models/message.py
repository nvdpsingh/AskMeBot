from pydantic import BaseModel, Field

class Message(BaseModel):
    username: str = Field(..., min_length=1)
    message: str = Field(...,min_length=1)