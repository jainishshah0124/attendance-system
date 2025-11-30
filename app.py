from flask import Flask,render_template,request,redirect,url_for, jsonify, session, abort, flash
from connection import engine
import sqlalchemy
import os
import face_recognition
import base64
import cv2
import numpy as np
import json
from google.cloud import storage
import io
from datetime import datetime, timedelta
import google.auth
from google.auth.transport.requests import Request
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sib_api_v3_sdk import Configuration, ApiClient, TransactionalEmailsApi, SendSmtpEmail


app = Flask(__name__,template_folder=os.getcwd()+'/Templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')


# Brevo (Sendinblue) transactional email configuration.
# Set BREVO_API_KEY/ BREVO_FROM_EMAIL / BREVO_FROM_NAME in the environment to enable notifications.
BREVO_API_KEY = os.environ.get('BREVO_API_KEY')
BREVO_FROM_EMAIL = os.environ.get('BREVO_FROM_EMAIL', 'no-reply@example.com')
BREVO_FROM_NAME = os.environ.get('BREVO_FROM_NAME', 'Attendance System')
brevo_api = None

if BREVO_API_KEY:
    brevo_config = Configuration()
    brevo_config.api_key['api-key'] = BREVO_API_KEY
    brevo_api = TransactionalEmailsApi(ApiClient(brevo_config))

def ensure_student_status_column():
    try:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SHOW COLUMNS FROM Student LIKE 'status'"))
            if result.fetchone() is None:
                conn.execute(sqlalchemy.text("ALTER TABLE Student ADD COLUMN status ENUM('pending','approved') NOT NULL DEFAULT 'pending'"))
                conn.commit()
    except Exception as exc:
        print(f"Unable to verify Student.status column: {exc}")

ensure_student_status_column()

def build_signed_url(photo_path):
    if not photo_path:
        return None
    if photo_path.startswith('http'):
        return photo_path
    if photo_path.startswith('gs://'):
        try:
            _, path = photo_path.split('gs://', 1)
            bucket_name, blob_name = path.split('/', 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            credentials, _ = google.auth.default()
            if not credentials.valid:
                credentials.refresh(Request())
            return blob.generate_signed_url(
                expiration=timedelta(minutes=30),
                credentials=credentials
            )
        except Exception as exc:
            print(f"Unable to sign URL for {photo_path}: {exc}")
            return None
    return photo_path

def send_student_status_email(email, name, approved, reason=None):
    if not brevo_api or not email:
        return
    status_text = "approved" if approved else "rejected"
    subject = f"Registration {status_text.title()}"
    body = (
        f"<p>Hi {name or 'Student'},</p>"
        f"<p>Your registration has been <strong>{status_text}</strong>.</p>"
        f"<p>{'You can now log in and link to classes.' if approved else (reason or 'Feel free to resubmit after correcting the details.')}</p>"
        "<p>– Attendance System</p>"
    )
    try:
        brevo_api.send_transac_email(SendSmtpEmail(
            to=[{"email": email, "name": name}],
            sender={"email": BREVO_FROM_EMAIL, "name": BREVO_FROM_NAME},
            subject=subject,
            html_content=body
        ))
    except Exception as exc:
        print(f"Failed to send status email to {email}: {exc}")

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            next_url = request.path if request.method == 'GET' else url_for('attendance')
            return redirect(url_for('login', next=next_url))
        return view_func(*args, **kwargs)
    return wrapper

def role_required(required_role):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if session.get('role') != required_role:
                return abort(403)
            return view_func(*args, **kwargs)
        return wrapper
    return decorator


@app.context_processor
def inject_user():
    return {
        "current_user": session.get('username'),
        "current_role": session.get('role')
    }


def get_system_stats():
    stats = {
        "students": 0,
        "teachers": 0,
        "classes": 0,
        "attendance_records": 0
    }
    with engine.connect() as conn:
        stats["students"] = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM Student")).scalar() or 0
        stats["classes"] = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM class")).scalar() or 0
        stats["attendance_records"] = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM attendance")).scalar() or 0
        stats["teachers"] = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM users WHERE role='teacher'")).scalar() or 0
    return stats


def get_recent_attendance(days=14):
    query = """
        SELECT record_date, status, COUNT(*) as count
        FROM attendance
        WHERE record_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
        GROUP BY record_date, status
        ORDER BY record_date DESC
    """
    analytics = {}
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(query), {"days": days}).fetchall()
        for row in rows:
            date_key = row.record_date.strftime("%Y-%m-%d")
            analytics.setdefault(date_key, {"Present": 0, "Absent": 0, "Late": 0})
            analytics[date_key][row.status.capitalize()] = int(row.count)
    return dict(sorted(analytics.items(), reverse=True))


def get_admin_reference_data():
    with engine.connect() as conn:
        classes = conn.execute(sqlalchemy.text("SELECT subject_code FROM class ORDER BY subject_code")).fetchall()
        teachers = conn.execute(sqlalchemy.text("SELECT id, username FROM users WHERE role='teacher' ORDER BY username")).fetchall()
        students = conn.execute(sqlalchemy.text("SELECT StudentID, fname, Lname FROM Student WHERE status='approved' ORDER BY fname")).fetchall()
        assignments = conn.execute(sqlalchemy.text("""
            SELECT tc.id, u.username, tc.subject_code
            FROM teacher_class tc
            JOIN users u ON u.id = tc.teacher_id
            ORDER BY u.username
        """)).fetchall()
    return {
        "classes": classes,
        "teachers": teachers,
        "students": students,
        "assignments": assignments
    }


def get_teacher_dashboard_data(teacher_id):
    data = {
        "classes": [],
        "students": [],
        "total_students": 0
    }
    with engine.connect() as conn:
        class_rows = conn.execute(sqlalchemy.text("""
            SELECT c.subject_code, c.start_time
            FROM teacher_class tc
            JOIN class c ON c.subject_code = tc.subject_code
            WHERE tc.teacher_id = :tid
            ORDER BY c.subject_code
        """), {"tid": teacher_id}).fetchall()

        all_students = conn.execute(sqlalchemy.text("""
            SELECT StudentID, fname, Lname
            FROM Student
            WHERE status='approved'
            ORDER BY fname
        """)).fetchall()

        for row in class_rows:
            enrolled = conn.execute(sqlalchemy.text("""
                SELECT s.StudentID, s.fname, s.Lname
                FROM enrollment e
                JOIN Student s ON s.StudentID = e.studentID
                WHERE e.subject_code = :code AND s.status='approved'
                ORDER BY s.fname
            """), {"code": row.subject_code}).fetchall()
            student_list = [
                {"id": stu.StudentID, "name": f"{stu.fname} {stu.Lname}"}
                for stu in enrolled
            ]
            data["classes"].append({
                "subject_code": row.subject_code,
                "start_time": row.start_time,
                "students": student_list,
                "student_count": len(student_list)
            })
            data["total_students"] += len(student_list)

    data["students"] = all_students
    return data


def get_attendance_classes(user_id, role):
    query = "SELECT subject_code, start_time FROM class ORDER BY subject_code"
    params = {}
    if role == 'teacher':
        query = """
            SELECT c.subject_code, c.start_time
            FROM teacher_class tc
            JOIN class c ON c.subject_code = tc.subject_code
            WHERE tc.teacher_id = :teacher_id
            ORDER BY c.subject_code
        """
        params["teacher_id"] = user_id

    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(query), params).fetchall()

    return [(row[0], row[1]) for row in rows]


def fetch_student_contacts(student_ids):
    if not student_ids:
        return {}
    placeholders = ",".join([":id{}".format(i) for i in range(len(student_ids))])
    params = {f"id{i}": sid for i, sid in enumerate(student_ids)}
    query = f"""
        SELECT StudentID, fname, Lname, email
        FROM Student
        WHERE StudentID IN ({placeholders})
    """
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(query), params).fetchall()
    contacts = {}
    for row in rows:
        contacts[int(row.StudentID)] = {
            "name": f"{row.fname} {row.Lname}",
            "email": row.email
        }
    return contacts


def send_attendance_notifications(records):
    if not brevo_api or not records:
        return
    student_ids = {int(rec["student_id"]) for rec in records}
    contacts = fetch_student_contacts(student_ids)
    for record in records:
        sid = int(record["student_id"])
        contact = contacts.get(sid)
        if not contact or not contact.get("email"):
            continue
        try:
            print(os.environ.get('BREVO_API_KEY'))
            record_date = record["record_date"]
            if hasattr(record_date, "strftime"):
                record_date_str = record_date.strftime("%Y-%m-%d")
            else:
                record_date_str = str(record_date)
            email = SendSmtpEmail(
                to=[{"email": contact["email"], "name": contact["name"]}],
                sender={"email": BREVO_FROM_EMAIL, "name": BREVO_FROM_NAME},
                subject=f"Attendance Update: {record['class_code']} on {record_date_str}",
                html_content=(
                    f"<p>Hi {contact['name']},</p>"
                    f"<p>Your attendance for <strong>{record['class_code']}</strong> on {record_date_str} "
                    f"was marked as <strong>{record['status'].title()}</strong>.</p>"
                    "<p>If you believe this is incorrect, please contact your instructor.</p>"
                    "<p>– Attendance System</p>"
                )
            )
            print(brevo_api.send_transac_email(email))
        except Exception as exc:
            print(f"Failed to send email for student {sid}: {exc}")


@app.route('/checkConn')
def checkConn():
    try:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT NOW();"))
            current_time = [row[0] for row in result][0]
        return f"Database connected successfully. Current DB time: {current_time}"
    except Exception as e:
        return f"Error connecting to database: {e}"

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('attendance'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        next_url = request.args.get('next') or url_for('attendance')

        if not username or not password:
            error = "Username and password are required."
        else:
            query = sqlalchemy.text("SELECT id, username, password_hash, role FROM users WHERE username=:username LIMIT 1")
            with engine.connect() as conn:
                row = conn.execute(query, {"username": username}).fetchone()
            if row and check_password_hash(row.password_hash, password):
                session['user_id'] = row.id
                session['username'] = row.username
                session['role'] = row.role
                return redirect(next_url)
            else:
                error = "Invalid username or password."

    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@login_required
@role_required('superadmin')
def admin_dashboard():
    stats = get_system_stats()
    analytics = get_recent_attendance()
    refs = get_admin_reference_data()
    return render_template(
        'admin_dashboard.html',
        stats=stats,
        analytics=analytics,
        classes=refs["classes"],
        teachers=refs["teachers"],
        students=refs["students"],
        assignments=refs["assignments"]
    )

@app.route('/teacher/dashboard')
@login_required
def teacher_dashboard():
    role = session.get('role')
    if role not in ('teacher', 'superadmin'):
        return abort(403)
    data = get_teacher_dashboard_data(session['user_id'])
    return render_template(
        'teacher_dashboard.html',
        classes=data["classes"],
        total_classes=len(data["classes"]),
        total_students=data["total_students"],
        students=data["students"]
    )

@app.route('/admin/classes', methods=['POST'])
@login_required
@role_required('superadmin')
def admin_create_class():
    subject_code = request.form.get('subject_code', '').strip()
    start_time = request.form.get('start_time', '').strip()
    if not subject_code or not start_time:
        flash("Subject code and start time are required.", "error")
        return redirect(url_for('admin_dashboard'))

    try:
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text("INSERT INTO class (subject_code, start_time) VALUES (:subject_code, :start_time)"),
                {"subject_code": subject_code, "start_time": start_time}
            )
        flash("Class created successfully.", "success")
    except Exception as exc:
        flash(f"Failed to create class: {exc}", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/teachers', methods=['POST'])
@login_required
@role_required('superadmin')
def admin_create_teacher():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for('admin_dashboard'))

    try:
        password_hash = generate_password_hash(password)
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text("INSERT INTO users (username, password_hash, role) VALUES (:username, :password_hash, 'teacher')"),
                {"username": username, "password_hash": password_hash}
            )
        flash("Teacher account created.", "success")
    except Exception as exc:
        flash(f"Failed to create teacher: {exc}", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/assign-teacher', methods=['POST'])
@login_required
@role_required('superadmin')
def admin_assign_teacher():
    teacher_id = request.form.get('teacher_id')
    subject_code = request.form.get('subject_code')
    if not teacher_id or not subject_code:
        flash("Teacher and class selection required.", "error")
        return redirect(url_for('admin_dashboard'))

    try:
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO teacher_class (teacher_id, subject_code)
                    VALUES (:teacher_id, :subject_code)
                    ON DUPLICATE KEY UPDATE subject_code = VALUES(subject_code)
                """),
                {"teacher_id": int(teacher_id), "subject_code": subject_code}
            )
        flash("Teacher assigned to class.", "success")
    except Exception as exc:
        flash(f"Failed to assign teacher: {exc}", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/assign-student', methods=['POST'])
@login_required
@role_required('superadmin')
def admin_assign_student():
    student_id = request.form.get('student_id')
    subject_code = request.form.get('subject_code')
    if not student_id or not subject_code:
        flash("Student and class selection required.", "error")
        return redirect(url_for('admin_dashboard'))

    try:
        with engine.connect() as conn:
            status_row = conn.execute(
                sqlalchemy.text("SELECT status FROM Student WHERE StudentID=:sid"),
                {"sid": int(student_id)}
            ).scalar()
            if status_row != 'approved':
                flash("Student must be approved before assigning.", "error")
                return redirect(url_for('admin_dashboard'))
        with engine.begin() as conn:
            exists = conn.execute(
                sqlalchemy.text("SELECT 1 FROM enrollment WHERE studentID=:sid AND subject_code=:code"),
                {"sid": int(student_id), "code": subject_code}
            ).fetchone()
            if exists:
                flash("Student already assigned to this class.", "info")
            else:
                conn.execute(
                    sqlalchemy.text("INSERT INTO enrollment (studentID, subject_code) VALUES (:sid, :code)"),
                    {"sid": int(student_id), "code": subject_code}
                )
                flash("Student assigned to class.", "success")
    except Exception as exc:
        flash(f"Failed to assign student: {exc}", "error")
    return redirect(url_for('admin_dashboard'))

def _json_response(query, params=None):
    params = params or {}
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(query), params).mappings().all()
        return [dict(row) for row in result]

@app.route('/admin/api/classes', methods=['GET'])
@login_required
@role_required('superadmin')
def api_list_classes():
    rows = _json_response("SELECT id, subject_code, start_time FROM class ORDER BY subject_code")
    return jsonify(rows)

@app.route('/admin/api/classes', methods=['POST'])
@login_required
@role_required('superadmin')
def api_create_class():
    data = request.get_json() or {}
    subject_code = data.get('subject_code', '').strip()
    start_time = data.get('start_time', '').strip()
    if not subject_code or not start_time:
        return jsonify({"error": "subject_code and start_time required"}), 400
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("INSERT INTO class (subject_code, start_time) VALUES (:subject_code, :start_time)"),
            {"subject_code": subject_code, "start_time": start_time}
        )
    return jsonify({"message": "class created"}), 201

@app.route('/admin/api/classes/<subject_code>', methods=['DELETE'])
@login_required
@role_required('superadmin')
def api_delete_class(subject_code):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM class WHERE subject_code=:code"), {"code": subject_code})
    return jsonify({"message": "class deleted"})

@app.route('/admin/api/teachers', methods=['GET'])
@login_required
@role_required('superadmin')
def api_list_teachers():
    rows = _json_response("SELECT id, username, role, created_at FROM users WHERE role='teacher' ORDER BY username")
    return jsonify(rows)

@app.route('/admin/api/teachers', methods=['POST'])
@login_required
@role_required('superadmin')
def api_create_teacher():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    password_hash = generate_password_hash(password)
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("INSERT INTO users (username, password_hash, role) VALUES (:username, :password_hash, 'teacher')"),
            {"username": username, "password_hash": password_hash}
        )
    return jsonify({"message": "teacher created"}), 201

@app.route('/admin/api/teachers/<int:teacher_id>', methods=['DELETE'])
@login_required
@role_required('superadmin')
def api_delete_teacher(teacher_id):
    with engine.begin() as conn:
        user = conn.execute(sqlalchemy.text("SELECT role FROM users WHERE id=:id"), {"id": teacher_id}).fetchone()
        if not user:
            return jsonify({"error": "teacher not found"}), 404
        if user.role != 'teacher':
            return jsonify({"error": "cannot delete non-teacher account"}), 400
        conn.execute(sqlalchemy.text("DELETE FROM users WHERE id=:id"), {"id": teacher_id})
    return jsonify({"message": "teacher deleted"})

@app.route('/admin/api/assignments', methods=['GET'])
@login_required
@role_required('superadmin')
def api_list_assignments():
    rows = _json_response("""
        SELECT tc.id, tc.teacher_id, tc.subject_code, u.username
        FROM teacher_class tc
        JOIN users u ON u.id = tc.teacher_id
        ORDER BY u.username
    """)
    return jsonify(rows)

@app.route('/admin/api/assignments', methods=['POST'])
@login_required
@role_required('superadmin')
def api_create_assignment():
    data = request.get_json() or {}
    teacher_id = data.get('teacher_id')
    subject_code = data.get('subject_code')
    if not teacher_id or not subject_code:
        return jsonify({"error": "teacher_id and subject_code required"}), 400
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("""
                INSERT INTO teacher_class (teacher_id, subject_code)
                VALUES (:teacher_id, :subject_code)
                ON DUPLICATE KEY UPDATE subject_code = VALUES(subject_code)
            """),
            {"teacher_id": int(teacher_id), "subject_code": subject_code}
        )
    return jsonify({"message": "assignment saved"}), 201

@app.route('/admin/api/assignments/<int:assignment_id>', methods=['DELETE'])
@login_required
@role_required('superadmin')
def api_delete_assignment(assignment_id):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM teacher_class WHERE id=:id"), {"id": assignment_id})
    return jsonify({"message": "assignment deleted"})

@app.route('/admin/api/student-assignments', methods=['POST'])
@login_required
@role_required('superadmin')
def api_assign_student_class():
    data = request.get_json() or {}
    student_id = data.get('student_id')
    subject_code = data.get('subject_code')
    if not student_id or not subject_code:
        return jsonify({"error": "student_id and subject_code required"}), 400
    with engine.begin() as conn:
        exists = conn.execute(
            sqlalchemy.text("SELECT 1 FROM enrollment WHERE studentID=:sid AND subject_code=:code"),
            {"sid": int(student_id), "code": subject_code}
        ).fetchone()
        if exists:
            return jsonify({"message": "student already assigned"}), 200
        conn.execute(
            sqlalchemy.text("INSERT INTO enrollment (studentID, subject_code) VALUES (:sid, :code)"),
            {"sid": int(student_id), "code": subject_code}
        )
    return jsonify({"message": "student assigned"}), 201

@app.route('/admin/api/student-assignments', methods=['DELETE'])
@login_required
@role_required('superadmin')
def api_delete_student_assignment():
    student_id = request.args.get('student_id')
    subject_code = request.args.get('subject_code')
    if not student_id or not subject_code:
        return jsonify({"error": "student_id and subject_code required"}), 400
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("DELETE FROM enrollment WHERE studentID=:sid AND subject_code=:code"),
            {"sid": int(student_id), "code": subject_code}
        )
    return jsonify({"message": "student assignment removed"})

@app.route('/teacher/assign-student', methods=['POST'])
@login_required
def teacher_assign_student():
    role = session.get('role')
    if role not in ('teacher', 'superadmin'):
        return abort(403)
    student_id = request.form.get('student_id')
    subject_code = request.form.get('subject_code')
    if not student_id or not subject_code:
        flash("Student and class selection required.", "error")
        return redirect(url_for('teacher_dashboard'))

    try:
        with engine.connect() as conn:
            status_row = conn.execute(
                sqlalchemy.text("SELECT status FROM Student WHERE StudentID=:sid"),
                {"sid": int(student_id)}
            ).scalar()
            if status_row != 'approved':
                flash("Student must be approved before linking.", "error")
                return redirect(url_for('teacher_dashboard'))
        with engine.begin() as conn:
            if role != 'superadmin':
                allowed = conn.execute(
                    sqlalchemy.text("""
                        SELECT 1 FROM teacher_class
                        WHERE teacher_id = :tid AND subject_code = :code
                    """),
                    {"tid": session['user_id'], "code": subject_code}
                ).fetchone()
                if not allowed:
                    flash("You are not assigned to this class.", "error")
                    return redirect(url_for('teacher_dashboard'))

            exists = conn.execute(
                sqlalchemy.text("SELECT 1 FROM enrollment WHERE studentID=:sid AND subject_code=:code"),
                {"sid": int(student_id), "code": subject_code}
            ).fetchone()
            if exists:
                flash("Student already linked to this class.", "info")
            else:
                conn.execute(
                    sqlalchemy.text("INSERT INTO enrollment (studentID, subject_code) VALUES (:sid, :code)"),
                    {"sid": int(student_id), "code": subject_code}
                )
                flash("Student linked successfully.", "success")
    except Exception as exc:
        flash(f"Failed to link student: {exc}", "error")
    return redirect(url_for('teacher_dashboard'))

@app.route('/api/verify-password', methods=['POST'])
@login_required
def verify_password():
    data = request.get_json(silent=True) or {}
    supplied = data.get('password')
    if not supplied:
        return jsonify({"valid": False, "message": "Password required"}), 400

    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text("SELECT password_hash FROM users WHERE id=:uid"),
            {"uid": session['user_id']}
        ).fetchone()
    if not row:
        return jsonify({"valid": False, "message": "User not found"}), 404

    if check_password_hash(row.password_hash, supplied):
        return jsonify({"valid": True})
    return jsonify({"valid": False, "message": "Incorrect password"}), 401

def can_review_students():
    return session.get('role') in ('teacher', 'superadmin')

@app.route('/pending_students')
@login_required
def pending_students():
    if not can_review_students():
        return abort(403)
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text("""
            SELECT StudentID, fname, Lname, gender, phone, email, address, photo_path, status
            FROM Student WHERE status='pending' ORDER BY fname
        """)).fetchall()

    students = []
    for row in rows:
        students.append({
            "id": row.StudentID,
            "name": f"{row.fname} {row.Lname}",
            "gender": row.gender,
            "phone": row.phone,
            "email": row.email,
            "address": row.address,
            "photo_url": build_signed_url(row.photo_path)
        })

    return render_template('pending_students.html', students=students)

@app.route('/pending_students/<int:student_id>/approve', methods=['POST'])
@login_required
def approve_student(student_id):
    if not can_review_students():
        return abort(403)
    reason = request.form.get('reason', '').strip()
    with engine.begin() as conn:
        student = conn.execute(
            sqlalchemy.text("SELECT fname, Lname, email FROM Student WHERE StudentID=:sid"),
            {"sid": student_id}
        ).fetchone()
        if not student:
            flash("Student not found.", "error")
            return redirect(url_for('pending_students'))
        conn.execute(
            sqlalchemy.text("UPDATE Student SET status='approved' WHERE StudentID=:sid"),
            {"sid": student_id}
        )
    send_student_status_email(student.email, f"{student.fname} {student.Lname}", True)
    flash("Student approved.", "success")
    return redirect(url_for('pending_students'))

@app.route('/pending_students/<int:student_id>/reject', methods=['POST'])
@login_required
def reject_student(student_id):
    if not can_review_students():
        return abort(403)
    with engine.begin() as conn:
        student = conn.execute(
            sqlalchemy.text("SELECT fname, Lname, email FROM Student WHERE StudentID=:sid"),
            {"sid": student_id}
        ).fetchone()
        if not student:
            flash("Student not found.", "error")
            return redirect(url_for('pending_students'))
        conn.execute(
            sqlalchemy.text("DELETE FROM Student WHERE StudentID=:sid"),
            {"sid": student_id}
        )
    send_student_status_email(student.email, f"{student.fname} {student.Lname}", False)
    flash("Student rejected and removed.", "info")
    return redirect(url_for('pending_students'))
@app.route('/register')
def register():
    return render_template('registration.html')

@app.route('/imageCapture',methods=['POST'])
def imageCapture():
    try:
        data = {
            "StudentID": int(request.form['employee_id']),
            "fname": request.form['first_name'],
            "Lname": request.form['last_name'],
            "DOB": request.form['dob'],
            "gender": request.form['gender'],
            "phone": request.form['phone'],
            "address": request.form['address'],
            "email": request.form['email'],
            "photo_path": ""  # You can fill this in later
        }

        print(data)

        insert_query = sqlalchemy.text("""
            INSERT INTO Student (StudentID, fname, Lname, DOB, gender, phone, address, email, photo_path, status)
            VALUES (:StudentID, :fname, :Lname, :DOB, :gender, :phone, :address, :email, :photo_path, 'pending')
        """)

        with engine.connect() as conn:
            conn.execute(insert_query, data)
            print("data inserted")
            conn.commit()

        return render_template('capture.html',employee_id=request.form['employee_id'])

    except Exception as e:
        return render_template('registration.html',error=str(e))
    
@app.route('/save_image',methods=['POST','GET'])
def save_image():
    image_data = request.form['imageData']
    employee_id = request.form['employee_id']

    # Decode base64 image (remove data URL prefix)
    _, encoded_data = image_data.split(',', 1)
    decoded_data = base64.b64decode(encoded_data)
    nparr = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid image data", 400

    # Normalize to RGB uint8 using shared helper
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    normalized_frame = ensure_rgb_uint8(rgb_frame)
    bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return "Failed to encode image", 500
    jpeg_bytes = buffer.tobytes()

    # Upload directly to GCS
    bucket_name = 'attendance--storage'
    destination_blob_name = f'{employee_id}.jpg'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Delete existing blob if it exists
    if blob.exists():
        blob.delete()

    # Upload the normalized JPEG from memory
    blob.upload_from_string(jpeg_bytes, content_type='image/jpeg')

    # Save GCS path in DB (or use blob.public_url if public)
    photo_path = f'gs://{bucket_name}/{destination_blob_name}'

    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("UPDATE Student SET photo_path=:path WHERE StudentID=:sid"),
            {"path": photo_path, "sid": int(employee_id)}
        )
        conn.commit()

    return redirect('/attendance')

@app.route('/submitAttendance', methods=['POST'])
@login_required
def submit_attendance():
    payload = request.get_json(silent=True) or {}
    attendance_records = payload.get('attendanceData') or []
    if not attendance_records:
        return jsonify({"message": "No attendance data provided"}), 400

    today_date = payload.get('todayDate')

    normalized_rows = []
    for record in attendance_records:
        try:
            student_id = record.get('rollNumber') or record.get('student_id')
            student_name = record.get('name')
            class_code = record.get('class') or payload.get('attendanceClass')
            status = record.get('status')
            record_date = record.get('date') or today_date
        except AttributeError:
            continue

        if not (student_id and student_name and class_code and status and record_date):
            continue

        try:
            parsed_date = datetime.strptime(record_date, "%Y-%m-%d").date()
        except ValueError:
            parsed_date = datetime.utcnow().date()

        normalized_rows.append({
            "student_id": str(student_id),
            "student_name": student_name,
            "class_code": class_code,
            "status": status,
            "record_date": parsed_date
        })

    if not normalized_rows:
        return jsonify({"message": "No valid attendance entries found"}), 400

    insert_stmt = sqlalchemy.text("""
        INSERT INTO attendance (student_id, student_name, class_code, status, record_date)
        VALUES (:student_id, :student_name, :class_code, :status, :record_date)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            updated_at = CURRENT_TIMESTAMP
    """)

    with engine.begin() as conn:
        conn.execute(insert_stmt, normalized_rows)

    send_attendance_notifications(normalized_rows)

    return jsonify({"message": "Attendance stored", "rows": len(normalized_rows)}), 200

@app.route('/add_class',methods=['POST'])
@login_required
def add_class():
    start_time = request.form['Time']
    subject_code = request.form['newClassName']

    try:
        with engine.connect() as conn:
            insert_query = sqlalchemy.text("""
                INSERT INTO class (start_time, subject_code)
                VALUES (:start_time, :subject_code)
            """)
            conn.execute(insert_query, {"start_time": start_time, "subject_code": subject_code})
            conn.commit()

        return redirect(url_for('attendance'))

    except Exception as e:
        # Optional: log the error or flash a message
        return f"Error adding class: {str(e)}", 500


def list_class_student():
    classes = []
    try:
        with engine.connect() as conn:
            query = sqlalchemy.text("SELECT subject_code, start_time FROM class")
            result = conn.execute(query)
            print(result)
            for row in result:
                classes.append(row[0])
                classes.append(row[1])

        return classes

    except Exception as e:
        print(f"Error fetching class data: {e}")
        return []


@app.route('/attendance')
@login_required
def attendance():
    role = session.get('role')
    user_id = session.get('user_id')
    class_rows = get_attendance_classes(user_id, role)

    listclass = [row[0] for row in class_rows]
    listTime = [row[1] for row in class_rows]

    student_payload = retrive_data(listclass) if listclass else {}

    localStorage_data = {
        "attendanceData": '[]',
        "classes": json.dumps(listclass),
        "students": json.dumps(student_payload),
        "listTime": json.dumps(listTime)
    }

    return render_template('attendance.html', localStorage_data=localStorage_data)

@app.route('/submit_fill_class',methods=['POST','GET'])
@login_required
def submit_fill_class():
    class_selected=request.form.getlist('classSelector')
    checked_students = request.form.getlist('students')
    rollList = request.form.getlist('localStorageData')[0].split(',')
    print('---------submit_fill_class starts---------')
    print(rollList)
    print(checked_students)
    print(class_selected)
    for each in checked_students:
        if each not in rollList:
            try:
                print('inside try')
                with engine.connect() as conn:
                    for each in checked_students:
                        if each not in rollList:
                            print(f"Inserting student {each} into class {class_selected[0]}")
                            insert_query = sqlalchemy.text("""
                                INSERT INTO enrollment (studentID, subject_code)
                                VALUES (:studentID, :class_code)
                            """)
                            conn.execute(insert_query, {"studentID": int(each), "class_code": class_selected[0]})

                    conn.commit()
            except Exception as e:
                print("Error adding class:" + str(e))
    print('Data inserted')
    
    temp = []
    for each in rollList:
        if each and each not in checked_students:
            try:
                print('inside try')
                with engine.connect() as conn:
                    delete_query = sqlalchemy.text("""
                        DELETE FROM enrollment
                        WHERE studentID = :student_id AND subject_code = :class_code
                    """)
                    result = conn.execute(delete_query, {
                        "student_id": int(each),
                        "class_code": class_selected[0]
                    })
                    print("deleted")
                    print(result.rowcount)
                    conn.commit()

                temp.append(int(each))

            except Exception as e:
                print("message :" + str(e))    
    data={
        "class_id":class_selected[0],
        "employee_id" : {"in":temp}
    }
    print(json.dumps(data))
    print('---------submit_fill_class ends---------')
    #responsedel = JSON.deleteJSONCALL(f'https://us-east-2.aws.neurelo.com/rest/enrollment_link?filter={json.dumps(data)}','','DELETE')
    return redirect(url_for('attendance'))

def retrive_data(classes):
    students_data = {}

    with engine.connect() as conn:
        for subject_code in classes:
            students_data[subject_code] = []

            # Get enrolled students for that classID
            enrollment_query = sqlalchemy.text("""
                SELECT s.StudentID, s.fname, s.Lname
                FROM enrollment e
                JOIN Student s ON e.studentID = s.StudentID
                WHERE e.subject_code = :subject_code AND s.status='approved'
            """)
            students = conn.execute(enrollment_query, {"subject_code": subject_code}).fetchall()

            for student in students:
                student_info = {
                    "name": f"{student[1]} {student[2]}",
                    "rollNumber": student[0]
                }
                students_data[subject_code].append(student_info)

    return students_data

def list_classes():
    classes = []
    try:
        with engine.connect() as conn:
            query = sqlalchemy.text("SELECT subject_code, start_time FROM class")
            result = conn.execute(query)
            print(result)
            for row in result:
                classes.append(row[0])
                classes.append(row[1])

        return classes

    except Exception as e:
        print(f"Error fetching class data: {e}")
        return []


@app.route('/fill_class')
@login_required
def fill_class():
    freshData = []

    try:
        with engine.connect() as conn:
            query = sqlalchemy.text("SELECT fname, Lname, StudentID FROM Student WHERE status='approved'")
            result = conn.execute(query)

            for row in result:
                temp = []
                temp.append(row[0])
                temp.append(row[1])
                temp.append(row[2])
                freshData.append(temp)

    except Exception as e:
        print(f"Error fetching student data: {e}")


    classes=list_classes()
    listclass=[]
    listTime=[]
    i=0
    for each in classes:
        if i%2==0:
            listclass.append(each)
        else:
            listTime.append(each)
        i=i+1
    print(listclass)
    print(json.dumps(retrive_data(listclass)));
    localStorage_data = {
        "classes": listclass,
        "colors": '{}',
        "debug": "honey:core-sdk:*",
        "students": json.dumps(retrive_data(listclass)),
        "toAssignstud" : freshData,
        "listTime":listTime
    }
    print(localStorage_data)
    return render_template('fill_class.html',localStorage_data=localStorage_data)

@app.route('/calendar')
@login_required
def calendar():
    # if 'username' not in session:
    #         error_message = "Please Login"
    #         return render_template('login.html', error_message=error_message)
    return render_template('cal.html')

def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert images to contiguous 8-bit RGB arrays for face_recognition."""
    print('Convert images to contiguous 8-bit RGB arrays for face_recognition.')
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    return np.require(image, dtype=np.uint8, requirements=['C_CONTIGUOUS', 'ALIGNED'])

@app.route('/handle_frameData',methods=['POST'])
@login_required
def handle_frameData():
    # print('start')
    data = request.json
    image_data = data.get('image_data')
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    dataset_path = 'Dataset/'
    bucket_name = 'attendance--storage'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT StudentID, photo_path FROM Student WHERE photo_path IS NOT NULL"))
        students = result.fetchall()


    for row in students:
        student_id = row[0]
        photo_path = row[1]  # e.g., gs://attendance-storage/123.jpg

        # Extract filename from path
        filename = os.path.basename(photo_path)
        blob = bucket.blob(filename)

        if not blob.exists():
            print(f"Image not found in bucket for {student_id}: {filename}")
            continue

        # Read blob data into memory
        img_bytes = blob.download_as_bytes()
        img = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to decode image for {student_id}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ensure_rgb_uint8(img)
        print(f"Loaded {filename} for {student_id}: type={type(img)}, shape={img.shape}, dtype={img.dtype}, flags={img.flags}")
        try:
            encodings = face_recognition.face_encodings(img)
        except RuntimeError as err:
            print(f"Skipping {student_id}: unable to encode face ({err})")
            continue
        if len(encodings) == 0:
            print(f"No face found in image for {student_id}")
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(str(student_id))

    #print('Face encoding complete')

    #retrive Data
    # for filename in os.listdir(dataset_path):
    #     file_path = os.path.join(dataset_path, filename)
    #     img = face_recognition.load_image_file(file_path)
    #     face_encoding = face_recognition.face_encodings(img)[0]
    #     if len(face_encoding) == 0:
    #         print(f"No face found in {filename}. Skipping.")
    #         continue
    #     known_face_encodings.append(face_encoding)
    #     name = os.path.splitext(filename)[0]
    #     known_face_names.append(name)
    
    # print('error')
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    # Receive frame data from client
    frame_data = image_data
    # Decode base64-encoded image data
    #image_data = base64.b64decode(frame_data.split(",")[1])
    _, encoded_data = frame_data.split(',', 1)
    decoded_data = base64.b64decode(encoded_data)
    
    # Convert decoded data to a NumPy array
    nparr = np.frombuffer(decoded_data, np.uint8)

    # Decode the NumPy array into an image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid frame data", 400

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = ensure_rgb_uint8(np.ascontiguousarray(small_frame[:, :, ::-1]))

    # Only process every other frame of video to save time
    print(f"Processing frame: shape={rgb_small_frame.shape}, dtype={rgb_small_frame.dtype}, contiguous={rgb_small_frame.flags['C_CONTIGUOUS']}")
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    except RuntimeError as err:
        print(f"Failed to process frame: {err}")
        return "Unable to process frame", 400

    face_names = []
    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        # print(name)
        face_names.append(name)
    return name

@app.route('/api/attendance/summary', methods=['GET'])
@login_required
def attendance_summary():
    class_code = request.args.get('class')
    month = request.args.get('month')  # format YYYY-MM

    base_query = """
        SELECT record_date as date, status, COUNT(*) as count
        FROM attendance
        WHERE 1=1
    """
    params = {}
    if class_code:
        base_query += " AND class_code = :class"
        params["class"] = class_code
    if month:
        base_query += " AND DATE_FORMAT(record_date, '%Y-%m') = :month"
        params["month"] = month

    base_query += " GROUP BY record_date, status ORDER BY record_date DESC"

    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(base_query), params)
        rows = [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "status": row.status,
                "count": int(row.count)
            }
            for row in result
        ]

    return jsonify({"data": rows})

@app.route('/api/attendance/details', methods=['GET'])
@login_required
def attendance_details():
    class_code = request.args.get('class')
    record_date = request.args.get('date')

    if not class_code or not record_date:
        return jsonify({"message": "class and date are required"}), 400

    query = """
        SELECT student_id, student_name, status
        FROM attendance
        WHERE class_code = :class AND record_date = :record_date
        ORDER BY student_name
    """
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(query), {"class": class_code, "record_date": record_date})
        records = [
            {
                "student_id": row.student_id,
                "student_name": row.student_name,
                "status": row.status
            }
            for row in result
        ]

    return jsonify({
        "class_code": class_code,
        "date": record_date,
        "records": records
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get('PORT',8080)))
