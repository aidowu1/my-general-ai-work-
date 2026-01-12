from sqlalchemy import or_, and_

from data_model import (create_existing_database_session, Teacher, Subject, Lesson)

session_instance = create_existing_database_session()

def get_teacher_name(teacher_id: int) -> str:
    """Retrieves the name of a teacher given their ID."""
    session = session_instance()
    teacher = session.query(Teacher).filter(Teacher.id == teacher_id).first()
    session.close()
    return teacher.name if teacher else "Unknown Teacher"

def get_all_lessons():
    """Retrieves all lessons in the database."""
    session = session_instance()
    results = session.query(Lesson).all()
    session.close()
    return results

def get_all_teachers():
    """Retrieves all teachers in the database."""
    session = session_instance()
    results = session.query(Teacher).all()
    session.close()
    return results

def get_all_subjects():
    """Retrieves all subjects in the database."""
    session = session_instance()
    results = session.query(Subject).all()
    session.close()
    return results  

def get_schedule(subject_type: str):
    """Retrieves the weekly timetable for a specific class."""
    session = session_instance()
    
    res = session.query(Lesson).join(Subject).filter(Subject.name == subject_type).all()
    session.close()
    return [f"{l.day_of_week}: {l.start_time} to {l.end_time} lesson delivered by {get_teacher_name(l.teacher_id)}" for l in res]

def get_schedule_v2(subject_type: str):
    """Retrieves the weekly timetable for a specific class."""
    session = session_instance()
    results = session.query(Lesson).join(Subject).filter(
        Subject.name == subject_type
    ).all()
    session.close()
    return results 

# Modify this query function to also return teacher names   
def get_time_schedule_by_teacher(teacher_name: str):
    """Retrieves lessons for a specific teacher with subject names."""
    session = session_instance()
    query = session.query(Lesson, Subject.name).join(Subject).join(Teacher)    
    results = query.filter(Teacher.name.contains(teacher_name.split()[-1])).all()
    session.close()
    raw_results = [f"Teacher: {teacher_name} is scheduled on {l.day_of_week} to deliver the subject: {subject_name} at {l.start_time} for {l.end_time}" 
                   for l, subject_name in results]
    return raw_results
    
def get_time_schedule_by_subject(subject_type: str):
    """Retrieves lessons for a specific subject."""
    session = session_instance()
    query = session.query(Lesson, Teacher.name).join(Teacher).join(Subject)
    results = query.filter(Subject.name == subject_type).all()
    session.close()
    # raw_results = [f"The subject: {subject_type} is scheduled on {l.day_of_week} to deliver by school teacher: {teacher_name} at {l.start_time} for {l.end_time}" 
    #                for l, teacher_name in results]
    raw_results = []
    for l, teacher_name in results:
        raw_results.append(
            f"The subject: {subject_type} is scheduled on {l.day_of_week} to deliver by school teacher: {teacher_name} at {l.start_time} for {l.end_time}"
        )
    return raw_results

def get_time_schedule_by_subject_and_teacher(subject_type: str, teacher_name: str):
    """Retrieves lessons for a specific subject and teacher."""
    session = session_instance()
    query = session.query(Lesson).join(Subject).join(Teacher)
    results = query.filter(
        Subject.name == subject_type,
        Teacher.name.contains(teacher_name.split()[-1])
    ).all()
    session.close()
    raw_results = [f"{l.day_of_week} at {l.start_time} for {l.end_time}" for l in results]
    return raw_results

def get_next_time_schedule(
        current_day: str, 
        current_time: str):
    """Retrieves the next lesson based on the current day and time."""
    session = session_instance()
    query = session.query(Lesson).join(Subject).join(Teacher)
    results = query.filter(
        or_(
            and_(
                Lesson.day_of_week == current_day,
                Lesson.start_time > current_time
            ),
            Lesson.day_of_week != current_day
        )
    ).order_by(Lesson.day_of_week, Lesson.start_time).all()
    session.close()
    raw_results = [f"{l.day_of_week} at {l.start_time} for {l.end_time}" for l in results]
    return raw_results


def get_next_time_schedule_by_teacher(
        teacher_name: str, 
        current_day: str, 
        current_time: str):
    """Retrieves the next lesson for a specific subject and teacher."""
    session = session_instance()
    query = session.query(Lesson).join(Subject).join(Teacher)
    results = query.filter(
        Teacher.name.contains(teacher_name.split()[-1]),
        or_(
            and_(
                Lesson.day_of_week == current_day,
                Lesson.start_time > current_time
            ),
            Lesson.day_of_week != current_day
        )
    ).order_by(Lesson.day_of_week, Lesson.start_time).all()
    session.close()
    raw_results = [f"{l.day_of_week} at {l.start_time} for {l.end_time}" for l in results]
    return raw_results


def get_next_time_schedule_by_subject(
        subject_type: str, 
        current_day: str, 
        current_time: str):
    """Retrieves the next lesson for a specific subject and teacher."""
    session = session_instance()
    query = session.query(Lesson).join(Subject).join(Teacher)
    results = query.filter(
        Subject.name == subject_type,        
        or_(
            and_(
                Lesson.day_of_week == current_day,
                Lesson.start_time > current_time
            ),
            Lesson.day_of_week != current_day
        )
    ).order_by(Lesson.day_of_week, Lesson.start_time).all()
    session.close()
    raw_results = [f"{l.day_of_week} at {l.start_time} for {l.end_time}" for l in results]
    return raw_results
    

tools = [get_schedule, 
         get_schedule_v2,
         get_time_schedule_by_teacher,
         get_time_schedule_by_subject,
         get_time_schedule_by_subject_and_teacher,
         get_next_time_schedule_by_teacher,
         get_next_time_schedule_by_subject]
