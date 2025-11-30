-- Students table stores student demographics and contact info (parent email used for notifications)
CREATE TABLE IF NOT EXISTS Student (
    StudentID INT AUTO_INCREMENT PRIMARY KEY,
    fname VARCHAR(255) NOT NULL,
    Lname VARCHAR(255) NOT NULL,
    DOB DATE,
    gender VARCHAR(32),
    phone VARCHAR(32),
    address VARCHAR(255),
    email VARCHAR(255),
    photo_path VARCHAR(512),
    status ENUM('pending','approved') NOT NULL DEFAULT 'pending'
);

-- Classes offered in the system
CREATE TABLE IF NOT EXISTS class (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_code VARCHAR(255) NOT NULL UNIQUE,
    start_time VARCHAR(64) NOT NULL
);

-- Teacher/administrator accounts
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('teacher','superadmin') NOT NULL DEFAULT 'teacher',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mapping between teachers and the classes they own
CREATE TABLE IF NOT EXISTS teacher_class (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id INT NOT NULL,
    subject_code VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_teacher_class (teacher_id, subject_code),
    CONSTRAINT fk_teacher FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_teacher_class FOREIGN KEY (subject_code) REFERENCES class(subject_code) ON DELETE CASCADE
);

-- Student enrollment per class
CREATE TABLE IF NOT EXISTS enrollment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    studentID INT NOT NULL,
    subject_code VARCHAR(255) NOT NULL,
    UNIQUE KEY uniq_student_class (studentID, subject_code),
    CONSTRAINT fk_enroll_student FOREIGN KEY (studentID) REFERENCES Student(StudentID) ON DELETE CASCADE,
    CONSTRAINT fk_enroll_class FOREIGN KEY (subject_code) REFERENCES class(subject_code) ON DELETE CASCADE
);

-- Attendance log per student/class/day
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(64) NOT NULL,
    student_name VARCHAR(255) NOT NULL,
    class_code VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL,
    record_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_attendance_entry (student_id, class_code, record_date)
);

-- Optional default super admin (adjust credentials before production)
INSERT INTO users (username, password_hash, role)
VALUES ('admin', '{bcrypt_hash_placeholder}', 'superadmin')
ON DUPLICATE KEY UPDATE username = username;
