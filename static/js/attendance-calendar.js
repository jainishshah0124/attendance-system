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
                if(data[i]['status']=='Present'){
                    present=parseInt(data[i]['count']);
                }
                else if(data[i]['status']=='Absent'){
                    absent=parseInt(data[i]['count']);
                }
                else if(data[i]['status']=='Late'){
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
async function retrieveAttendanceSummary() {
    const subjectCode = document.getElementById('classSelector').value; // Assuming you have an input field for subject code
    try {
        debugger;
        const data = {"data":[{"date":"2025-03-21"},{"date":"2025-03-10"},
        {
            date: "2025-03-25",
            status: "Present",
            count: "18"
          },
          {
            date: "2025-03-25",
            status: "Absent",
            count: "3"
          },
          {
            date: "2025-03-25",
            status: "Late",
            count: "2"
          },
          {
            date: "2025-03-24",
            status: "Present",
            count: "20"
          },
          {
            date: "2025-03-24",
            status: "Absent",
            count: "1"
          }]};
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

function onPopup(day){
    localStorage.setItem('attendanceClass',document.getElementById("classSelector").value);
    if(day<10){
        day='0'+day;
    }
    localStorage.setItem('clickedBtn',localStorage.getItem("currMonYear")+"-"+day);
    var popup = document.getElementById('linkStudentPopup');
    popup.classList.add('fade-in'); // Add class to trigger fade-in animation
    popup.classList.remove('fade-out');
    popup.style.display = 'block';
    document.getElementById('popupFrame').contentWindow.retriveStudentList();
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