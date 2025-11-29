const date = new Date();

const renderCalendar = () => {
  date.setDate(1);
  const monthDays = document.querySelector(".days");
  const infoContainer = document.querySelector(".info");

  const lastDay = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
  const firstDayIndex = date.getDay();
  const lastDayIndex = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDay();
  const nextDays = 7 - lastDayIndex - 1;

  const months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];

  document.querySelector(".month h1").innerHTML = months[date.getMonth()];
  document.querySelector(".month p").innerHTML = new Date().toDateString();
  var temp=parseInt(date.getMonth()+1);
  if(temp<10){
    temp="0"+temp;
  }
  localStorage.setItem('currMonYear',date.getFullYear()+"-"+temp);

  let days = "";

  for (let x = firstDayIndex; x > 0; x--) {
    days += `<div class="prev-date">${new Date(date.getFullYear(), date.getMonth(), -x + 1).getDate()}</div>`;
  }
  var temp = JSON.parse(localStorage.getItem('attendanceData'));
  console.log(temp);
  var totalDays=[];
  for(var i=0;i<temp.length;i++){
    var mon=parseInt(temp[i]['date'].split('-')[1]);
    var dt=parseInt(temp[i]['date'].split('-')[2]);
    if((date.getMonth()+1)==mon){
        totalDays.push(dt);
    }
  }
  for (let i = 1; i <= lastDay; i++) {
    if(totalDays.includes(i)){
        days += `<div class="highlighted" onClick="onPopup(${i})">${i}</div>`;
    }else{
        days += `<div>${i}</div>`;
    }
  }

  for (let j = 1; j <= nextDays; j++) {
    days += `<div class="next-date">${j}</div>`;
  }
  monthDays.innerHTML = days;

  const dayElements = document.querySelectorAll(".days div:not(.prev-date):not(.next-date)");

  dayElements.forEach(dayElement => {
    dayElement.addEventListener("mouseenter", () => {
        var day = dayElement.innerText;
        const month = months[date.getMonth()];
        const year = date.getFullYear();
        var mon;
        if(date.getMonth()+1<10){
            mon='0'+(date.getMonth()+1);
        }
        else{
            mon=date.getMonth()+1;
        }
        if(day<10){
            day='0'+day;
        }
        mon=year+'-'+mon+'-'+day;
        var present=0;var late=0;var absent=0;var total=0;
        var data = JSON.parse(localStorage.getItem('attendanceData'));
        for(var i=0;i<data.length;i++){
            if(data[i]['date']==mon){
                if(data[i]['status']=='present'){
                    present=parseInt(data[i]['count']);
                }
                else if(data[i]['status']=='absent'){
                    absent=parseInt(data[i]['count']);
                }
                else if(data[i]['status']=='late'){
                    late=parseInt(data[i]['count']);
                }
            }
          }
      total=present+late+absent;
      let temp1={total:total,present:present,absent:absent,late:late}
      localStorage.setItem('summary',JSON.stringify(temp1));
      if(total>0){
        const info = `Total Students: ${total} | Present: ${present} | Absent: ${absent} | Late: ${late}`;
        infoContainer.innerText = info;
      }
      
    });  
    dayElement.addEventListener("mouseleave", () => {
        infoContainer.innerText = "";
      });  
  });
};

document.querySelector(".prev").addEventListener("click", () => {
  date.setMonth(date.getMonth() - 1);
  renderCalendar();
});

document.querySelector(".next").addEventListener("click", () => {
  date.setMonth(date.getMonth() + 1);
  renderCalendar();
});

window.onload = function() {
    const display = document.getElementById('clock');

    function updateTime() {
        const date = new Date();

        const hour = formatTime(date.getHours());
        const minutes = formatTime(date.getMinutes());
        const seconds = formatTime(date.getSeconds());



        display.innerText=`${hour} : ${minutes} : ${seconds}`
    }

    function formatTime(time) {
        if ( time < 10 ) {
            return '0' + time;
        }
        return time;
    }
    // Update the live time initially
    updateTime();
    setInterval(updateTime, 1000);
};
function populateClasses() {
    // Retrieve classes from local storage
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
// async function retrieveAttendanceSummary() {
//     const subjectCode = document.getElementById('classSelector').value; // Assuming you have an input field for subject code
//     try {
//         debugger;
//         const data = {"data":[{"date":"2025-03-21"},{"date":"2025-03-10"},
//         {
//             date: "2025-11-25",
//             status: "Present",
//             count: "18"
//           },
//           {
//             date: "2025-11-25",
//             status: "Absent",
//             count: "3"
//           },
//           {
//             date: "2025-11-25",
//             status: "Late",
//             count: "2"
//           },
//           {
//             date: "2025-11-24",
//             status: "Present",
//             count: "20"
//           },
//           {
//             date: "2025-11-24",
//             status: "Absent",
//             count: "1"
//           }]};
//         console.log(data); // Do something with the response data
//         console.log('Data sent successfully:', data.data);
//         localStorage.setItem('attendanceData',JSON.stringify(data.data));
//         renderCalendar();
//         return data;
//     } catch (error) {
//         console.error('There was a problem with the fetch operation:', error);
//     }
// }

async function retrieveAttendanceSummary() {
    const subjectCode = document.getElementById('classSelector').value; // Assuming you have an input field for subject code
    try {
        const monthKey = localStorage.getItem('currMonYear');
        const response = await fetch(`/api/attendance/summary?class=${encodeURIComponent(subjectCode)}&month=${monthKey}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        console.log(data); // Do something with the response data
        console.log('Data sent successfully:', data.data);
        localStorage.setItem('attendanceData',JSON.stringify(data.data));
        renderCalendar();
        return data;
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
    }
}


function callMethod(){
    retrieveAttendanceSummary();
}
document.addEventListener("DOMContentLoaded",
    function () {
        populateClasses();
        callMethod();
    });

async function onPopup(day){
    const classCode = document.getElementById("classSelector").value;
    localStorage.setItem('attendanceClass', classCode);
    if(day<10){
        day='0'+day;
    }
    const dateKey = localStorage.getItem("currMonYear")+"-"+day;
    localStorage.setItem('clickedBtn', dateKey);

    const popup = document.getElementById('linkStudentPopup');
    popup.classList.add('fade-in');
    popup.classList.remove('fade-out');
    popup.style.display = 'block';

    setPopupLoading(classCode, dateKey);
    try {
        const response = await fetch(`/api/attendance/details?class=${encodeURIComponent(classCode)}&date=${dateKey}`);
        if (!response.ok) {
            throw new Error('Failed to load attendance details');
        }
        const data = await response.json();
        populatePopup(data, classCode, dateKey);
    } catch (error) {
        console.error(error);
        showPopupError(error.message);
    }
}

function setPopupLoading(classCode, dateKey) {
    document.getElementById('popupTitle').innerText = `${classCode} - ${dateKey}`;
    document.getElementById('popupSubtitle').innerText = 'Loading attendance...';
    document.getElementById('popupTotal').innerText = 'Total: 0';
    document.getElementById('popupPresent').innerText = 'Present: 0';
    document.getElementById('popupAbsent').innerText = 'Absent: 0';
    document.getElementById('popupLate').innerText = 'Late: 0';
    document.getElementById('popupList').innerHTML = '<p class="popup-empty">Loading...</p>';
}

function showPopupError(message) {
    document.getElementById('popupSubtitle').innerText = message;
    document.getElementById('popupList').innerHTML = '<p class="popup-empty">Unable to load attendance details.</p>';
}

function populatePopup(data, classCode, dateKey) {
    const title = document.getElementById('popupTitle');
    const subtitle = document.getElementById('popupSubtitle');
    const list = document.getElementById('popupList');

    title.innerText = `${classCode} - ${dateKey}`;

    const records = data.records || [];
    if (records.length === 0) {
        subtitle.innerText = 'No attendance recorded for this day.';
        list.innerHTML = '<p class="popup-empty">No entries found.</p>';
        updatePopupSummary(0, 0, 0, 0);
        return;
    }

    subtitle.innerText = `${records.length} record(s)`;
    const counts = {present: 0, absent: 0, late: 0};
    list.innerHTML = records.map((record) => {
        const statusKey = normalizeStatus(record.status);
        if (counts.hasOwnProperty(statusKey)) {
            counts[statusKey] += 1;
        }
        const statusLabel = statusDisplay(statusKey);
        return `<div class="popup-list-item">
                    <div>
                        <div class="student-name">${record.student_name}</div>
                        <div class="student-id">ID: ${record.student_id}</div>
                    </div>
                    <span class="status-pill status-${statusKey}">${statusLabel}</span>
                </div>`;
    }).join('');

    const total = counts.present + counts.absent + counts.late;
    updatePopupSummary(total, counts.present, counts.absent, counts.late);
}

function updatePopupSummary(total, present, absent, late) {
    document.getElementById('popupTotal').innerText = `Total: ${total}`;
    document.getElementById('popupPresent').innerText = `Present: ${present}`;
    document.getElementById('popupAbsent').innerText = `Absent: ${absent}`;
    document.getElementById('popupLate').innerText = `Late: ${late}`;
}

function normalizeStatus(status) {
    if (!status) return 'absent';
    const value = status.toLowerCase();
    if (value === 'leave') return 'late';
    return ['present', 'late', 'absent'].includes(value) ? value : 'absent';
}

function statusDisplay(statusKey) {
    switch (statusKey) {
        case 'present':
            return 'Present';
        case 'late':
            return 'Late';
        default:
            return 'Absent';
    }
}

function exportPopupToPDF() {
    const popup = document.querySelector('#linkStudentPopup .popup-content');
    if (!popup) {
        return;
    }
    const printWindow = window.open('', '_blank', 'width=800,height=600');
    const styles = document.querySelector('link[href*="attendance-calendar.min.css"]');
    printWindow.document.write(`
        <html>
        <head>
            <title>Attendance Summary</title>
            ${styles ? `<link rel="stylesheet" href="${styles.href}">` : ''}
            <style>
                body { font-family: "Quicksand", sans-serif; padding: 20px; background: #fff; color: #000; }
                .popup-content { color: #000; }
                .popup-list-item { border-color: #ccc; }
                .status-present, .status-absent, .status-late { color: #000 !important; }
            </style>
        </head>
        <body>${popup.innerHTML}</body>
        </html>
    `);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
}
function closePopup() {
    // Close the currently open popup
    var popupIds = ['linkStudentPopup'];

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
