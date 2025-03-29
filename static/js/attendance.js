//script.js
function data() {
    debugger;
    for (const [key, value] of Object.entries(localStorageData)) {
        localStorage.setItem(key, value);
    }
}
function showAddStudentForm() {
    var popup = document.getElementById('addStudentPopup');
    popup.classList.add('fade-in'); // Add class to trigger fade-in animation
    popup.classList.remove('fade-out');
    popup.style.display = 'block';
}

function showlinkStudentForm() {
    var popup = document.getElementById('linkStudentPopup');
    popup.classList.add('fade-in'); // Add class to trigger fade-in animation
    popup.classList.remove('fade-out');
    popup.style.display = 'block';
}

function handleFormSubmission(event, iframe_name, form_name) {
    event.preventDefault(); // Prevent the default form submission behavior

    // Get the form element from the iframe
    var iframe = document.getElementById(iframe_name);
    var form = iframe.contentWindow.document.getElementById(form_name);
    //imageForm

    if (form == null) {
        iframe = document.getElementById('popupregFrame');
        form = iframe.contentWindow.document.getElementById('imageForm');
    }

    // Submit the form within the iframe
    form.submit();

    // Close the popup
    closePopup();

    // Redirect the whole page
    window.location.href = '/attendance';
}



function showAddClassForm() {
    var popup = document.getElementById('addClassPopup');
    popup.classList.add('fade-in'); // Add class to trigger fade-in animation
    popup.classList.remove('fade-out');
    popup.style.display = 'block';
}

function addStudent() {
    // Get input values
    const newStudentName = document.
        getElementById('newStudentName').value;
    const newStudentRoll = document.
        getElementById('newStudentRoll').value;

    if (!newStudentName || !newStudentRoll) {
        alert("Please provide both name and roll number.");
        return;
    }

    // Add the new student to the list
    const classSelector = document.
        getElementById('classSelector');
    const selectedClass = classSelector.
        options[classSelector.selectedIndex].value;
    const studentsList = document.
        getElementById('studentsList');

    const listItem = document.createElement('li');
    listItem.setAttribute('data-roll-number', newStudentRoll);
    listItem.innerHTML =
        `<strong>
            ${newStudentName}
        </strong> 
        <div class="status" style="flex-grow:1;font-size:large;font-weight:bolder;color:darkblue;"></div>
        (Roll No. ${newStudentRoll})`;

    const absentButton =
        createButton('A', 'absent',
            () => markAttendance('absent', listItem, selectedClass));
    const presentButton =
        createButton('P', 'present',
            () => markAttendance('present', listItem, selectedClass));
    const leaveButton =
        createButton('L', 'leave',
            () => markAttendance('leave', listItem, selectedClass));

    listItem.appendChild(absentButton);
    listItem.appendChild(presentButton);
    listItem.appendChild(leaveButton);

    studentsList.appendChild(listItem);
    saveStudentsList(selectedClass);
    closePopup();
}

function addClass() {
    const newClassName = document.
        getElementById('newClassName').value;

    if (!newClassName) {
        alert("Please provide a class name.");
        return;
    }

    // Add the new class to the class selector
    const classSelector = document.
        getElementById('classSelector');
    const newClassOption = document.
        createElement('option');
    newClassOption.value = newClassName;
    newClassOption.text = newClassName;
    classSelector.add(newClassOption);
    saveClasses();
    closePopup();
}

function submitAttendance() {
    if(localStorage.getItem('attendanceData')==null){
        alert('Nothing to submit');
        return;
    }
    const classSelector = document.
        getElementById('classSelector');

    if (!classSelector || !classSelector.options ||
        classSelector.selectedIndex === -1) {
        console.error
            ('Class selector is not properly defined or has no options.');
        return;
    }

    const selectedClass = classSelector.
        options[classSelector.selectedIndex].value;

    if (!selectedClass) {
        console.error('Selected class is not valid.');
        return;
    }

    const studentsList =
        document.getElementById('studentsList');

    // Check if attendance is already submitted 
    // for the selected class
    var attendanceData = localStorage.getItem('attendanceData');
    var attendanceClass = localStorage.getItem('attendanceClass');
    const now = new Date();
    const date =
        `${now.getFullYear()}-${String(now.getMonth() + 1).
            padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    $.ajax({
        url: '/submitAttendance',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ attendanceData: JSON.parse(attendanceData), todayDate: date, attendanceClass: JSON.parse(attendanceClass) }),
        success: function (response) {
            console.log('Data sent successfully:', response);
        },
        error: function (xhr, status, error) {
            console.error('Error sending data:', error);
        }
    });
    const isAttendanceSubmitted =
        isAttendanceSubmittedForClass(selectedClass);

    if (isAttendanceSubmitted) {
        // If attendance is submitted, hide the 
        // summary and show the attendance result
        document.getElementById('summarySection').
            style.display = 'none';
        showAttendanceResult(selectedClass);
    } else {
        // If attendance is not submitted, show the summary
        document.getElementById('summarySection').
            style.display = 'block';
        document.getElementById('resultSection').
            style.display = 'none';
    }
    // Clear the student list and reset the form
    studentsList.innerHTML = '';
}

function isAttendanceSubmittedForClass(selectedClass) {
    const savedAttendanceData = JSON.parse
        (localStorage.getItem('attendanceData')) || [];
    return savedAttendanceData.some
        (record => record.class === selectedClass);
}

function showAttendanceResult(selectedClass) {
    const resultSection = document.
        getElementById('resultSection');

    if (!resultSection) {
        console.error('Result section is not properly defined.');
        return;
    }

    const now = new Date();
    const date =
        `${now.getFullYear()}-${String(now.getMonth() + 1).
            padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const time =
        `${String(now.getHours()).padStart(2, '0')}:
        ${String(now.getMinutes()).padStart(2, '0')}:
        ${String(now.getSeconds()).padStart(2, '0')}`;

    // Retrieve attendance data from local storage
    const savedAttendanceData = JSON.parse
        (localStorage.getItem('attendanceData')) || [];
    const filteredAttendanceData = savedAttendanceData.
        filter(record => record.class === selectedClass);

    const totalStudents = filteredAttendanceData.
        reduce((acc, record) => {
            if (!acc.includes(record.name)) {
                acc.push(record.name);
            }
            return acc;
        }, []).length;

    const totalPresent = filteredAttendanceData.
        filter(record => record.status === 'present').length;
    const totalAbsent = filteredAttendanceData.
        filter(record => record.status === 'absent').length;
    const totalLeave = filteredAttendanceData.
        filter(record => record.status === 'leave').length;

    // Update the result section
    document.getElementById('attendanceDate').
        innerText = date;
    document.getElementById('attendanceTime').
        innerText = time;
    document.getElementById('attendanceClass').
        innerText = selectedClass;
    document.getElementById('attendanceTotalStudents').
        innerText = totalStudents;
    document.getElementById('attendancePresent').
        innerText = totalPresent;
    document.getElementById('attendanceAbsent').
        innerText = totalAbsent;
    document.getElementById('attendanceLeave').
        innerText = totalLeave;

    // Show the attendance result section
    resultSection.style.display = 'block';
}

function closePopup() {
    // Close the currently open popup
    var popupIds = ['addStudentPopup', 'addClassPopup', 'linkStudentPopup'];

    // Loop through each popup ID
    popupIds.forEach(function (id) {
        var popup = document.getElementById(id);

        // Add fade-out animation class
        popup.classList.add('fade-out');

        // Hide the popup after animation completes
        setTimeout(function () {
            popup.style.display = 'none';
        }, 500); // Duration of fade-out animation

        // Remove fade-in class
        popup.classList.remove('fade-in');
    });
}

function createButton(text, status, onClick) {
    const button = document.createElement('button');
    button.type = 'button';
    button.innerText = text;
    button.className = status;
    button.onclick = onClick;
    //button.style="display:none;"
    return button;
}

function populateClasses() {
    // Retrieve classes from local storage
    debugger;
    const savedClasses = JSON.parse
        (localStorage.getItem('classes')) || [];
    const classSelector =
        document.getElementById('classSelector');

    savedClasses.forEach(className => {
        const newClassOption =
            document.createElement('option');
        newClassOption.value = className;
        newClassOption.text = className;
        classSelector.add(newClassOption);
    });
}

function showStudentsList() {
     document.getElementById('classTime').innerHTML='Class Timings:' + JSON.parse(localStorage.getItem('listTime'))[JSON.parse(localStorage.getItem('classes')).indexOf(document.getElementById('classSelector').value)];
    //  document.getElementById('classTime').innerHTML='Class Timings: 18:00';
    localStorage.removeItem('attendanceData');
    localStorage.removeItem('colors');
    const classSelector =
        document.getElementById('classSelector');
    const selectedClass = classSelector.
        options[classSelector.selectedIndex].value;

    const studentsList =
        document.getElementById('studentsList');
    studentsList.innerHTML = '';

    // Retrieve students from local storage
    const savedStudents = JSON.parse
        (localStorage.getItem('students')) || {};
    const selectedClassStudents =
        savedStudents[selectedClass] || [];

    selectedClassStudents.forEach(student => {
        if (student.name == undefined)
            return;
        const listItem = document.createElement('li');
        listItem.setAttribute
            ('data-roll-number', student.rollNumber);
        listItem.innerHTML =
            `<strong>
                ${student.name}
            </strong> 
            <div class="status" style="flex-grow:1;font-size:large;font-weight:bolder;color:darkblue;"></div>
            (Roll No. ${student.rollNumber})`;

        const absentButton = createButton('A', 'absent',
            () => markAttendance('absent', listItem, selectedClass));
        const presentButton = createButton('P', 'present',
            () => markAttendance('present', listItem, selectedClass));
        const leaveButton = createButton('L', 'leave',
            () => markAttendance('leave', listItem, selectedClass));
        // absentButton.disabled=true;
        // presentButton.disabled=true;
        // leaveButton.disabled=true;
        const savedColor = getSavedColor
            (selectedClass, student.rollNumber);
        if (savedColor) {
            listItem.style.backgroundColor = savedColor;
        }

        listItem.appendChild(absentButton);
        listItem.appendChild(presentButton);
        listItem.appendChild(leaveButton);

        studentsList.appendChild(listItem);
    });

    // Check if attendance for the 
    // selected class has been submitted
    const resultSection = document.
        getElementById('resultSection');
    const isAttendanceSubmitted =
        resultSection.style.display === 'block';

    // Show the appropriate section based 
    // on the attendance submission status
    if (isAttendanceSubmitted) {
        // Attendance has been submitted, 
        // show the attendance result
        showAttendanceResult(selectedClass);
    } else {
        // Attendance not submitted, 
        // show the normal summary
        showSummary(selectedClass);
    }
}

function showAttendanceResult(selectedClass) {
    const resultSection =
        document.getElementById('resultSection');

    if (!resultSection) {
        console.error('Result section is not properly defined.');
        return;
    }

    const now = new Date();
    const date =
        `${now.getFullYear()}-${String(now.getMonth() + 1).
            padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const time =
        `${String(now.getHours()).padStart(2, '0')}:
        ${String(now.getMinutes()).padStart(2, '0')}:
        ${String(now.getSeconds()).padStart(2, '0')}`;

    // Retrieve attendance data from local storage
    const savedAttendanceData = JSON.parse
        (localStorage.getItem('attendanceData')) || [];
    const filteredAttendanceData = savedAttendanceData.
        filter(record => record.class === selectedClass);

    const totalStudents = filteredAttendanceData.
        reduce((acc, record) => {
            if (!acc.includes(record.name)) {
                acc.push(record.name);
            }
            return acc;
        }, []).length;

    const totalPresent = filteredAttendanceData.
        filter(record => record.status === 'present').length;
    const totalAbsent = filteredAttendanceData.
        filter(record => record.status === 'absent').length;
    const totalLeave = filteredAttendanceData.
        filter(record => record.status === 'leave').length;

    // Update the result section
    const resultContent =
        `Date: ${date} | Time: ${time} | 
        Total Students: ${totalStudents} | 
        Present: ${totalPresent} | 
        Absent: ${totalAbsent} | Late: ${totalLeave}`;
    resultSection.innerHTML = resultContent;

    // Show the result section
    resultSection.style.display = 'block';

    // Show the list of students below the result section
    const studentsListHTML =
        generateStudentsListHTML(filteredAttendanceData);
    resultSection.insertAdjacentHTML
        ('afterend', studentsListHTML);
}


function markAttendance
    (status, listItem, selectedClass) {
    // Find the corresponding student name
    const studentName = listItem.
        querySelector('strong').innerText;

    // Update the background color of the student 
    // row based on the attendance status
    listItem.style.backgroundColor =
        getStatusColor(status);

    // Save the background color to local storage
    saveColor(selectedClass,
        listItem.getAttribute('data-roll-number'),
        getStatusColor(status));

    // Update the attendance record for the specific student
    updateAttendanceRecord(studentName, listItem.getAttribute('data-roll-number'), selectedClass, status);

    // Show the summary for the selected class
    showSummary(selectedClass);
}

function getStatusColor(status) {
    switch (status) {
        case 'absent':
            return '#e74c3c';
        case 'present':
            return '#2ecc71';
        case 'leave':
            return '#f39c12';
        case 'late':
            return '#f39c12';
        default:
            return '';
    }
}

function updateAttendanceRecord
    (studentName, rollNum, selectedClass, status) {
    // Retrieve existing attendance data from local storage
    const savedAttendanceData = JSON.parse
        (localStorage.getItem('attendanceData')) || [];

    // Check if the record already exists
    const existingRecordIndex = savedAttendanceData.
        findIndex(record => record.name === studentName &&
            record.class === selectedClass);

    if (existingRecordIndex !== -1) {
        // Update the existing record
        savedAttendanceData[existingRecordIndex].
            status = status;
        savedAttendanceData[existingRecordIndex].
            date = getCurrentDate();
    } else {
        // Add a new record
        savedAttendanceData.push(
            {
                name: studentName, class: selectedClass,
                status: status, date: getCurrentDate(), rollNumber: rollNum
            });
    }

    // Save updated attendance data to local storage
    localStorage.setItem('attendanceData',
        JSON.stringify(savedAttendanceData));
    localStorage.setItem('attendanceClass',
        JSON.stringify(selectedClass));
}

function showSummary(selectedClass) {
    const savedAttendanceData = JSON.parse
        (localStorage.getItem('attendanceData')) || [];

    // Filter attendance data based on the selected class
    const filteredAttendanceData = savedAttendanceData.
        filter(record => record.class === selectedClass);

    const totalStudents = filteredAttendanceData.
        reduce((acc, record) => {
            if (!acc.includes(record.name)) {
                acc.push(record.name);
            }
            return acc;
        }, []).length;

    const totalPresent = filteredAttendanceData.
        filter(record => record.status === 'present').length;
    const totalAbsent = filteredAttendanceData.
        filter(record => record.status === 'absent').length;
    const totalLeave = filteredAttendanceData.
        filter(record => record.status === 'leave').length;

    document.getElementById('totalStudents').
        innerText = totalStudents;
    document.getElementById('totalPresent').
        innerText = totalPresent;
    document.getElementById('totalAbsent').
        innerText = totalAbsent;
    document.getElementById('totalLeave').
        innerText = totalLeave;
}

function getCurrentDate() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).
        padStart(2, '0');
    const day = String(now.getDate()).
        padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function saveClasses() {
    // Save classes to local storage
    const classSelector = document.
        getElementById('classSelector');
    const savedClasses = Array.from(classSelector.options).
        map(option => option.value);
    localStorage.setItem('classes',
        JSON.stringify(savedClasses));
}

function saveStudentsList(selectedClass) {
    // Save the updated student list to local storage
    const studentsList = document.
        getElementById('studentsList');
    const savedStudents = JSON.parse
        (localStorage.getItem('students')) || {};
    const selectedClassStudents = Array.from(studentsList.children).
        map(item => {
            return {
                name: item.querySelector('strong').innerText,
                rollNumber: item.getAttribute('data-roll-number')
            };
        });

    savedStudents[selectedClass] = selectedClassStudents;
    localStorage.setItem
        ('students', JSON.stringify(savedStudents));
}

function saveColor(selectedClass, rollNumber, color) {
    const savedColors = JSON.parse
        (localStorage.getItem('colors')) || {};
    if (!savedColors[selectedClass]) {
        savedColors[selectedClass] = {};
    }
    savedColors[selectedClass][rollNumber] = color;
    localStorage.setItem('colors',
        JSON.stringify(savedColors));
}

function getSavedColor(selectedClass, rollNumber) {
    const savedColors = JSON.parse
        (localStorage.getItem('colors')) || {};
    return savedColors[selectedClass] ?
        savedColors[selectedClass][rollNumber] : null;
}

// Get the modal
var modal = document.getElementById("myModal");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("modal-close")[0];

// When the user clicks the button, open the modal

function handleRestrictedButtonClick(callingLogic) {
    // Get the modal
    var modal = document.getElementById("myModal");
  
    // Show the modal
    modal.style.display = "block";
  
    // Handle submit button click
    var submitButton = document.getElementById("submitButton");
    var existingListeners = submitButton.cloneNode(true);
    submitButton.parentNode.replaceChild(existingListeners, submitButton);
    document.getElementById("submitButton").addEventListener("click", function() {
        handleSubmitButtonClick(callingLogic);
    });
        var closeButton = document.getElementsByClassName("modal-close")[0];
    closeButton.onclick = function() {
        modal.style.display = "none";
    };

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
        if (event.target == modal) {
        modal.style.display = "none";
        }
    };

  }

  function handleSubmitButtonClick(callingLogic) {
    var modal = document.getElementById("myModal");
    // Get the password input field
    var password = document.getElementById("passwordInput").value;
    // Check if password is correct (e.g., compare with session password)
    if (password === "admin") {
      modal.style.display = "none";
      passwordInput.value="";
      callingLogic();
    } else {
      console.log("Incorrect password");
      // Highlight the input field to indicate incorrect password
      passwordInput.classList.add('incorrect-password');
      var modalContent = document.getElementsByClassName("modal-content")[0];
      // display an error message
      var errorMessage = document.createElement('p');
      errorMessage.textContent = 'Incorrect password. Please try again.';
      errorMessage.classList.add('error-message');
      modalContent.appendChild(errorMessage);
      // clear password input field after a delay
      setTimeout(function() {
          passwordInput.value = "";
          passwordInput.classList.remove('incorrect-password');
          errorMessage.remove();
          }, 2000); // Change delay as needed
      }
  }
