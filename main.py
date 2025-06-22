from fastapi import FastAPI, UploadFile, HTTPException, Depends,status
from pydantic import BaseModel
from typing import Annotated
from  models import Base,DocumentDetail,InterviewBooking
from database import engine, Sessionlocal
from sqlalchemy.orm import Session
from agent import Agent
from datetime import datetime
from email_service import EmailService

app = FastAPI()
Base.metadata.create_all(bind=engine)
 
class DocumentDetailBase(BaseModel):
    id: int
    filename: str
    chunking_method: str
    embedding_model_name: str
    embedding_provider: str


class InterviewBookingBase(BaseModel):
    id: int
    full_name: str
    email: str
    interview_date: str
    interview_time: str
    booking_time: str

class QueryRequest(BaseModel):
    query: str
    session_id: str

def get_database():
    db = Sessionlocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_database)]


@app.post("/uploadfile")
async def upload_file(file:UploadFile,chunking_method:str,embedding_model_name:str,embedding_provider:str,db:db_dependency):
    try:
        print(embedding_model_name)
        if not file.filename.endswith(("txt","pdf")):
            raise HTTPException(status_code=400,detail="Please upload valid format(pdf or text).")

        contents = await file.read()
        
        data = DocumentDetail(
            filename = file.filename,
            chunking_method = chunking_method,
            embedding_model = embedding_model_name,
            embedding_provider = embedding_provider
        )
        db.add(data)
        db.commit()
        

        result = Agent(file.filename, contents, chunking_method, embedding_model_name, embedding_provider)
        chunk_counts,embeding_latency,retrieval_latency, result = result.process_file()
        return {
        "result": result,
        "chunk_count": chunk_counts,
        "embedding_latency": embeding_latency,
        "retrieval_latency": retrieval_latency        
            }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        


@app.post("/userquery")
async def documents_query(query:QueryRequest):
    try:
        agent = Agent(contents=None,chunking_method=None,embedding_model_name=None,embedding_provider=None)
        response = await agent.process_query(query=query.query,session_id=query.session_id or "default")
        
        return {
            "answer": response['answer'],
            "sources": response['results'],
            "session_id": response['session_id']             
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interviewbooking")
async def interview_booking(request: InterviewBookingBase, db:db_dependency):
    try:

        booking = InterviewBooking(
            full_name=request.full_name,
            email=request.email,
            interview_date=request.interview_date,
            interview_time=request.interview_time,
            booking_time=datetime.today()
        )
        db.add(booking)
        db.commit()

        email_service = EmailService()
        await email_service.send_confirmation_email(
            booking.full_name,
            booking.email,
            booking.interview_date,
            booking.interview_time,
            booking.booking_time
        )
        
        return {
            "message": "Interview booked successfully",
            "booking_id": booking.id,
            "confirmation_sent": True
        }
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


