from data_model import (create_database_session, Teacher, SchoolClass, Lesson)
session_instance = create_database_session()

def get_teacher_name(teacher_id: int) -> str:
    """Retrieves the name of a teacher given their ID."""
    session = session_instance()
    teacher = session.query(Teacher).filter(Teacher.id == teacher_id).first()
    session.close()
    return teacher.name if teacher else "Unknown Teacher"

def get_schedule(class_name: str):
    """Retrieves the weekly timetable for a specific class."""
    session = session_instance()
    
    res = session.query(Lesson).join(SchoolClass).filter(SchoolClass.grade_name == class_name).all()
    session.close()
    return [f"{l.day_of_week}: {l.start_time} to {l.end_time} lesson delivered by {get_teacher_name(l.teacher_id)}" for l in res]

def get_schedule_v2(class_name: str):
    """Retrieves the weekly timetable for a specific class."""
    session = session_instance()
    results = session.query(Lesson).join(SchoolClass).filter(
        SchoolClass.grade_name == class_name
    ).all()
    session.close()
    return results   
    

tools = [get_schedule, get_schedule_v2]
