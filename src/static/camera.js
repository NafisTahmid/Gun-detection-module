document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Fetch the JSON data from the server
        const response = await fetch('/server_config', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        console.log(response.statusText);  // Debugging the response

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log(data);
        const tableBody = document.querySelector('#camera_table tbody');
        // const numberOfColumns = tableBody.querySelectorAll('tr:first-child td').length;

        // Clear any existing rows
        tableBody.innerHTML = '';

        if (data.cameras && data.cameras.length > 0) {
            data.cameras.forEach(camera => {
                const row = document.createElement('tr');

                // Create and append Camera ID cell
                const idCell = document.createElement('td');
                idCell.innerText = camera.camera_id;
                row.appendChild(idCell);

                const url = camera.camera_url;
                let ipAddress = "";
                let port = "";

                // Check if the URL contains '@' (indicating user credentials are present)
                if (url.includes('@')) {
                    const urlParts = url.split('@')[1]; // Get the part after '@'
                    if (urlParts) {
                        const [hostPart] = urlParts.split('/'); // Get the part before '/'
                        [ipAddress, port] = hostPart.split(':'); // Split IP and port
                    }
                } else {
                    // If no '@' is found, split from '://' and get the host part
                    const urlParts = url.split('://')[1];
                    if (urlParts) {
                        const [hostPart] = urlParts.split('/'); // Get the part before '/'
                        [ipAddress, port] = hostPart.split(':'); // Split IP and port
                    }
                }

                // Create and append Address cell
                const addressCell = document.createElement('td');
                addressCell.innerText = port ? `${ipAddress}:${port}` : ipAddress;
                row.appendChild(addressCell);


                // Camera Type
                const typeCell = document.createElement('td');
                typeCell.innerText = camera.camera_type;
                typeCell.classList.add('capitalize');
                row.appendChild(typeCell);

                // Create and append Camera Status cell
                const statusCell = document.createElement('td');
                statusCell.innerText = camera.camera_running_status ? 'Active' : 'Inactive';
                statusCell.style.color = camera.camera_running_status ? 'green' : 'red';
                statusCell.style.fontWeight = 'bold';
                row.appendChild(statusCell);

                // Could add buttons here
                // Add edit button
                const editButtonCell = document.createElement("td");
                const editButton = document.createElement("button");
                editButton.innerText = "Edit";
                editButton.style.backgroundColor = (camera.camera_running_status ?  "#ffffcc" : "#FFC000");
                editButton.disabled = camera.camera_running_status ? true : false;
                editButton.addEventListener("click", () => openEditForm(camera));
                editButtonCell.appendChild(editButton);
                row.appendChild(editButtonCell);

                // Add delete button
                const deleteButtonCell = document.createElement("td");
                const deleteButton = document.createElement("button");
                deleteButton.innerText = "Delete";
                deleteButton.style.backgroundColor = (camera.camera_running_status ? "#ffb3b3" : "#c70009");
                deleteButton.disabled = camera.camera_running_status ? true : false;
                deleteButton.addEventListener("click", () => deleteCamera(camera.camera_id));
                deleteButtonCell.appendChild(deleteButton);
                row.appendChild(deleteButtonCell);

                // Stop camera button
                const cameraToggleButtonCell = document.createElement("td");

                const stopCameraButton = document.createElement("button");
                stopCameraButton.style.backgroundColor = "#c70009";
                stopCameraButton.innerText = "Stop camera";
                stopCameraButton.addEventListener("click", () => stopCamera(camera.camera_id));
    
                // Start camera button
                const startCameraButton = document.createElement("button");
                startCameraButton.style.backgroundColor = "#32de84";
                startCameraButton.innerText = "start camera"
                startCameraButton.addEventListener("click", () => startCamera(camera.camera_id));
               

                // Append either stop or start button
                cameraToggleButtonCell.appendChild(camera.camera_running_status ? stopCameraButton : startCameraButton);
                row.appendChild(cameraToggleButtonCell);

                // Append the row to the table body
                tableBody.appendChild(row);
            });
        } else {
            // Create a row that spans all columns with a message
            const noCam = document.getElementById('error_message');
            // Clear any existing rows
            tableBody.innerHTML = '';

            noCam.innerHTML = 'No Camera Found.. Handle From Frontend';
            noCam.style.textAlign = 'center'; // Center the text
            noCam.style.fontStyle = 'italic'; // Italicize the text
            // Add border and padding styles
            noCam.style.border = '2px solid #172554'; // Add a red border
            noCam.style.padding = '2px'; // Add 2px padding
        }
    } catch (error) {
        console.error('Error fetching JSON:', error);
        // Handle the error case with a message
        const tableBody = document.querySelector('#camera_table tbody');
        const errorBody = document.getElementById('error_message');
        // Clear any existing rows
        tableBody.innerHTML = '';

        errorBody.innerHTML = 'No Camera Installed';
        errorBody.style.textAlign = 'center'; // Center the text
        errorBody.style.fontStyle = 'italic'; // Italicize the text
        // Add border and padding styles
        errorBody.style.border = '2px solid #172554'; // Add a red border
        errorBody.style.padding = '2px'; // Add 2px padding
    }
});

async function deleteCamera(camera_id) {
    if (confirm("Are you sure you want to delete the camera?")) {
        try {
            const response = await fetch(`/cameras/${camera_id}`, {
                method: "DELETE"
            });
            if (!response.ok) {
                throw new Error("Failed to delete camera");

            }
            alert("Camera deleted successfully");
            fetchCameras();
            setTimeout(() => location.reload(), 500);
        } catch(error) {
            console.error("Error deleting camera: ", error);
            // alert("Failed to delete camera. Check the console for details");
        }
    }
}

function openEditForm(camera) {
    // Populate the form with camera details
    document.getElementById("camera_id").value = camera.camera_id;
    document.getElementById("camera_url").value = camera.camera_url;
    document.getElementById("camera_type").value = camera.camera_type;
    document.getElementById("camera_running_status").checked = camera.camera_running_status;
    document.getElementById("threshold").value = camera.threshold;
    document.getElementById("third_party").checked = camera.third_party;

    // Show the form
    const open_edit_form = document.getElementById("open_edit_form");
    open_edit_form.style.display = "block";
    open_edit_form.style.top = "50%";
    open_edit_form.style.transform = "translate(-50%, -50%)";

    let overlay = document.getElementById("overlay");
    overlay.style.display = "block";
}

async function editCamera(event) {
    // Prevent the default form submission behavior
    event.preventDefault();

    // Get the camera_id from the input field
    const camera_id = parseInt(document.getElementById("camera_id").value);
    const camera_url = document.getElementById("camera_url").value;
    const camera_type = document.getElementById("camera_type").value;
    const camera_running_status = document.getElementById("camera_running_status").checked; // Use `.checked` for checkbox
    const threshold = parseFloat(document.getElementById("threshold").value);
    const third_party = document.getElementById("third_party").checked; // Use `.checked` for checkbox

    // Form data to be sent in the PUT request
    const formData = {
        "camera_id": camera_id,
        "camera_url": camera_url,
        "camera_type": camera_type,
        "camera_running_status": camera_running_status,
        "threshold": threshold,
        "third_party": third_party
    };

    try {
        // Make the API request to update the camera
        const response = await fetch(`/cameras/${camera_id}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error("API call failed");
        }

       

        // Reload the page after 1 second to reflect the changes
        setTimeout(() => location.reload(), 500);
    } catch (error) {
        console.error("Error editing camera: ", error);
    }
}

function closePopUp() {
    const open_edit_form = document.getElementById("open_edit_form");
    open_edit_form.style.display = "none";
    open_edit_form.style.top = "0";
    open_edit_form.style.transform = "translate(-50%, -50%) scale(0.1)";
    open_edit_form.style.cursor = "pointer";

    let overlay = document.getElementById("overlay");
    overlay.style.display = "none";

}

async function stopCamera(camera_id) {
    try {
        const response = await fetch(`/cameras/${camera_id}/stop_thread`, {
            method: "POST"
        })
        if(!response.ok) {
            throw new Error("Failed to stop camera");
        }
        console.log("Camera stopped");
        setTimeout(() => location.reload(), 200);
    } catch(error) {
        console.error("Error message: ", error);
    }
}

async function startCamera(camera_id) {
    try {
        const response = await fetch(`/cameras/${camera_id}/start`, {
            method: "POST"
        })
        if (!response.ok) {
            throw new Error("Error starting camera");
        }
        console.log("Camera successfully stopped");
        setTimeout(() => location.reload(), 200);
    } catch(error) {
        alert("Failed to start camera");
        console.error("Error message: ", error);
    }
}

// Example of calling openEditForm when an edit button is clicked
document.getElementById("edit-form").addEventListener("submit", editCamera);