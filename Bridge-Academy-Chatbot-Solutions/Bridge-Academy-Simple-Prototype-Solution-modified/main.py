from data_model import (create_new_database_session, Teacher, Subject, Lesson)
import tools
import configs


def create_chatbot_session() -> None:
    """
    Create and return a new database session for chatbot interactions.
    """    
    Session = create_new_database_session()
    session = Session()
    return session

def create_session_for_existing_db(db_path: str=configs.DB_PATH) -> None:
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

def check_timetable_data_inserts() -> None:
    """
    Check and print all lessons in the database to verify data inserts.
    """
    lessons = tools.get_all_lessons()
    if not lessons:
        print("No lessons found in the database.")
    else:       
        for l in lessons:
            print(f"Lesson: {l.day_of_week} {l.start_time}-{l.end_time} SubjectID:{l.subject_id} TeacherID:{l.teacher_id}")
    
    teachers = tools.get_all_teachers()
    if not teachers:
        print("No lessons found in the database.")
    else:    
        for t in teachers:
            print(f"Teacher: ID:{t.id} Name:{t.name} SubjectID:{t.subject_id}")

    subjects = tools.get_all_subjects()
    if not subjects:
        print("No lessons found in the database.")
    else:
        for s in subjects:
            print(f"Subject: ID:{s.id} Name:{s.name}")
        

if __name__ == "__main__":
    # Example usage
    db_session = create_chatbot_session()
    check_timetable_data_inserts()