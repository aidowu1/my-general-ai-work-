from data_model import (create_database_session, Teacher, SchoolClass, Lesson)
import tools

DB_PATH = 'sqlite:///school_timetable.db'

def create_chatbot_session() -> None:
    """
    Create and return a new database session for chatbot interactions.
    """    
    Session = create_database_session()
    session = Session()
    return session

def create_session_for_existing_db(db_path: str=DB_PATH) -> None:
    """
    Create and return a new database session for an existing database.
    param: db_path: The file path to the existing database.
    """    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def check_schedules(session) -> None:
    """
    Example function to check and print all lessons in the database.
    param: session: The database session to use for queries.
    """
    class_name = "10-Alpha"
    print(f"Checking all lessons for the specified class name {class_name}:")
    lessons = tools.get_schedule(class_name) 
    for lesson in lessons:
        # print(f"Lesson ID: {lesson.id}, Class ID: {lesson.class_id}, Teacher ID: {lesson.teacher_id}, Day: {lesson.day_of_week}, Start: {lesson.start_time}, End: {lesson.end_time}")
        print(lesson)

if __name__ == "__main__":
    # Example usage
    db_session = create_chatbot_session()
    # db_ession = create_session_for_existing_db()
    # You can now use db_session to interact with the database
    check_schedules(db_session)