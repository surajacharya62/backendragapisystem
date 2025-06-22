from sqlalchemy import Column, Integer, String, DateTime, Text
from database import Base

class DocumentDetail(Base):
    __tablename__ = "document_details"
    
    id = Column(Integer,primary_key=True,index=True)
    filename = Column(String, index=True, unique=True)   
    chunking_method = Column(String(100))
    embedding_model = Column(String(100))
    embedding_provider = Column(String(50))    
   

class InterviewBooking(Base):
    __tablename__ = "booking_details"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100))
    email = Column(String(150))
    interview_date = Column(String(50))
    interview_time = Column(String(50))
    booking_time = Column(DateTime)