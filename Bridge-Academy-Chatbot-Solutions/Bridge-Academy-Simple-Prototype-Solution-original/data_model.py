from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Time
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class Teacher(Base):
    __tablename__ = 'teachers'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    subject = Column(String)

class SchoolClass(Base):
    __tablename__ = 'classes'
    id = Column(Integer, primary_key=True)
    grade_name = Column(String) # e.g., "10A"

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True)
    class_id = Column(Integer, ForeignKey('classes.id'))
    teacher_id = Column(Integer, ForeignKey('teachers.id'))
    day_of_week = Column(String)
    start_time = Column(Time)
    end_time = Column(Time)

def create_database_session() -> sessionmaker:
    """
    Create a new database session and return the sessionmaker object.
    Returns:
        sessionmaker: A configured sessionmaker object for database interactions.
    """
    engine = create_engine('sqlite:///school_timetable.db')
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session
