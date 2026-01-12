import datetime
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Time, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import configs

Base = declarative_base()

class Subject(Base):
    __tablename__ = 'subjects'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    lessons = relationship("Lesson", back_populates="subject")
    
class Teacher(Base):
    __tablename__ = 'teachers'
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'))
    name = Column(String)    
    lessons = relationship("Lesson", back_populates="teacher")

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'))
    teacher_id = Column(Integer, ForeignKey('teachers.id'))
    day_of_week = Column(String) # e.g., "Monday"
    start_time = Column(Time)
    end_time = Column(Time)
    
    teacher = relationship("Teacher", back_populates="lessons")
    subject = relationship("Subject", back_populates="lessons")   

def create_new_database_session() -> sessionmaker:
    """
    Create a new database session and return the sessionmaker object.
    Returns:
        sessionmaker: A configured sessionmaker object for database interactions.
    """
    engine = create_engine(configs.DB_PATH)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session

def create_existing_database_session() -> sessionmaker:
    """
    Create a new database session for an existing database and return the sessionmaker object.
    Returns:
        sessionmaker: A configured sessionmaker object for database interactions.
    """
    engine = create_engine(configs.DB_PATH)
    session = sessionmaker(bind=engine)
    return session
