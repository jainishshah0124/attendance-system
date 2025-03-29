 document.addEventListener("DOMContentLoaded",
        function () {
            const loadingScreen = document.getElementById("loading-screen");
            const content = document.getElementById("content");

            // Check if webcam video frame is loaded
            const webcamVideo = document.getElementById("videoElement");

            webcamVideo.addEventListener("loadeddata", function() {
                // Hide loading screen once webcam video frame is loaded
                document.getElementById("")
                loadingScreen.style.display = "none";
            });

            data();
            populateClasses();
            showStudentsList();
        });
        const videoElement = document.getElementById("videoElement");
        const statusMessage = document.getElementById("statusMessage");
        let websocket;
        let sendFrameInterval;

        function disableEnableAllButtons(){
            var buttons = document.querySelectorAll('button');
            // Loop through each button and disable it
            if(document.getElementById('startSocket').style.display=='block'){
                document.getElementById('startSocket').style="display:none";
                document.getElementById('stopSocket').style="display:block";
                buttons.forEach(function(button) {
                    if(button.id!='stopSocket' && button.id!='submitButton'){
                        //button.disabled = true;
                    }
                });
                document.getElementById('classSelector').disabled=true;
            }else{
                //websocket.close();
                clearInterval(sendFrameInterval);
                document.querySelectorAll('li').forEach(function(temp) {
                    // Do something with each 'li' element
                    if(temp.attributes.style==undefined){
                        var rollNumber=temp.dataset.rollNumber;
                        const button = document.querySelector(`li[data-roll-number="${rollNumber}"] button.absent`);
                        button.click();
                        receiveMessageFromWebSocket(rollNumber,'Absent Marked');
                        console.log(rollNumber); 
                    }
                });
                document.getElementById('startSocket').style="display:block";
                document.getElementById('stopSocket').style="display:none";
                buttons.forEach(function(button) {
                    if(button.id!='stopSocket'){
                        //button.disabled = false;
                    }
                });
                document.getElementById('classSelector').disabled=false;
            }
        }

        // Function to establish WebSocket connection
        function connectToWebSocket() {
            const dockerContainerIP = '127.0.0.1';
            websocket = new WebSocket(`ws://${dockerContainerIP}:8767`);  // Adjust the WebSocket server URL and port

            // Event handler for when the WebSocket connection is opened
            websocket.onopen = function(event) {
                console.log("WebSocket connection established.");
                
                websocket.onmessage = function(event) {
                    var dbTime=addMinutesToTime(JSON.parse(localStorage.getItem('listTime'))[JSON.parse(localStorage.getItem('classes')).indexOf(document.getElementById('classSelector').value)].toString(),parseInt(document.getElementById('bufferTime').value));
                    const rollNumber=event.data;
                    if(document.querySelector(`li[data-roll-number="${rollNumber}"]`).style.backgroundColor==''){
                        var curr=document.getElementById('clock').innerHTML.split(':')[0].trim() + ':' + document.getElementById('clock').innerHTML.split(':')[1].trim();
                        if(compareTimes(dbTime,curr)==-1){
                            const button = document.querySelector(`li[data-roll-number="${rollNumber}"] button.leave`);
                            button.disabled=false;
                            button.click();
                            button.disabled=true;
                            receiveMessageFromWebSocket(rollNumber,'Late Marked')
                        }else{
                            const button = document.querySelector(`li[data-roll-number="${rollNumber}"] button.present`);
                            button.disabled=false;
                            button.click();
                            button.disabled=true;
                            receiveMessageFromWebSocket(rollNumber,'Present Marked')
                        }
                    }
                 };
            };

            // Event handler for when the WebSocket connection encounters an error
            websocket.onerror = function(event) {
                console.error("WebSocket error:", event);
            };

            // Event handler for when the WebSocket connection is closed
            websocket.onclose = function(event) {
                console.log("WebSocket connection closed.");
                clearInterval(sendFrameInterval);
            };
        }
        
        function addMinutesToTime(time, minutesToAdd) {
            const [hours, minutes] = time.split(":").map(Number);
            const totalMinutes = hours * 60 + minutes;
            const newTotalMinutes = totalMinutes + minutesToAdd;
            const newHours = Math.floor(newTotalMinutes / 60) % 24;
            const newMinutes = newTotalMinutes % 60;
            const newTime = `${String(newHours).padStart(2, "0")}:${String(newMinutes).padStart(2, "0")}`;
            return newTime;
        }

        function compareTimes(time1, time2) {
            const [hours1, minutes1] = time1.split(":").map(Number);
            const [hours2, minutes2] = time2.split(":").map(Number);
            if (hours1 < hours2) {
                return -1; // time1 is earlier
            } else if (hours1 > hours2) {
                return 1; // time2 is earlier
            } else {
                // If hours are equal, compare minutes
                if (minutes1 < minutes2) {
                    return -1; // time1 is earlier
                } else if (minutes1 > minutes2) {
                    return 1; // time2 is earlier
                } else {
                    return 0; // times are equal
                }
            }
        }




        function receiveMessageFromWebSocket(message,status) {
            var popup = document.createElement('div');
            popup.textContent = status;
            popup.classList.add('popup_message');

            document.querySelector(`li[data-roll-number="${message}"] div.status`).appendChild(popup);

            setTimeout(function() {
                            popup.remove();
                        }, 1500);
        }

        // Function to capture frame, encode it, and send it to WebSocket server
        async function sendFrameToServer() {
            const canvas = document.createElement("canvas");
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const context = canvas.getContext("2d");
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

            // Convert canvas data to Base64-encoded image
            const imageData = canvas.toDataURL("image/jpeg");

            // Send image data to WebSocket server
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(imageData);
            } else {
                console.error("WebSocket connection is not open.");
            }
        }

        

        async function startVideoStream() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoElement.srcObject = stream;
            } catch (error) {
                console.error("Error accessing camera:", error);
            }
        }


        function sendFrameToFlask() {
            const canvas = document.createElement("canvas");
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const context = canvas.getContext("2d");
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

            // Convert canvas data to Base64-encoded image
            const imageData = canvas.toDataURL("image/jpeg");

            fetch('/handle_frameData', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_data: imageData }),
              })
              .then(response => {
                if (!response.ok) {
                  throw new Error('Network response was not ok');
                }
                return response.json();
              })
              .then(data => {
                console.log('Response:', data);
                var dbTime=addMinutesToTime(JSON.parse(localStorage.getItem('listTime'))[JSON.parse(localStorage.getItem('classes')).indexOf(document.getElementById('classSelector').value)].toString(),parseInt(document.getElementById('bufferTime').value));
                    const rollNumber=data;
                    if(document.querySelector(`li[data-roll-number="${rollNumber}"]`).style.backgroundColor==''){
                        var curr=document.getElementById('clock').innerHTML.split(':')[0].trim() + ':' + document.getElementById('clock').innerHTML.split(':')[1].trim();
                        if(compareTimes(dbTime,curr)==-1){
                            const button = document.querySelector(`li[data-roll-number="${rollNumber}"] button.leave`);
                            button.disabled=false;
                            button.click();
                            button.disabled=true;
                            receiveMessageFromWebSocket(rollNumber,'Late Marked');
                        }else{
                            const button = document.querySelector(`li[data-roll-number="${rollNumber}"] button.present`);
                            button.disabled=false;
                            button.click();
                            button.disabled=true;
                            receiveMessageFromWebSocket(rollNumber,'Present Marked');
                        }
                    }
              })
              .catch(error => {
                console.error('Error:', error);
              });
        }

        // Connect to WebSocket when page loads
        window.onload = function() {
            startVideoStream();
            document.getElementById('addStudentBtn').addEventListener('click', function() {
                window.location.href = '/register'; // Redirect to the /register endpoint
            });
            document.getElementById('startSocket').addEventListener('click', function() {
                //connectToWebSocket();
                sendFrameInterval=setInterval(sendFrameToFlask, 2000);
                disableEnableAllButtons();
                // Repeat sending frames to server every 5 seconds
                //sendFrameInterval=setInterval(sendFrameToServer, 1500);
            });
            document.getElementById('stopSocket').addEventListener('click', function() {
                handleRestrictedButtonClick(disableEnableAllButtons);
                
            });
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