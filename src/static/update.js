// const token = 'y_1htLH2vs3My9CtGXk5';
const token = 'zFhaVbHzbbGkWpijdXH5';
const repoApiUrl = 'https://gitlab.accelx.net/api/v4/projects/152/repository/branches';
const commitsApiUrl = 'https://gitlab.accelx.net/api/v4/projects/152/repository/commits?ref_name=API-Server-ML';
let currentVersion = "";

const loadBranchesBtn = document.getElementById('loadBranchesBtn');

async function fetchVersion() {
    try {
        const response = await fetch('/check_version', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }

        });

        if (!response.ok) {
            throw new Error(`Error fetching version info: ${response.status} - ${response.statusText}`);
        }
        const data = await response.json();
        console.log(data)
        currentVersion = data.message;
        console.log("Current version:", currentVersion);

    } catch (error) {
        console.error("Failed to fetch version:", error.message);
    }
}



// Function to fetch the latest commits from the API-Server-ML branch
async function fetchCommits() {
    try {
        // Add spinner
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = "";
        loadBranchesBtn.classList.add('loading');
        loadBranchesBtn.disabled = true;

        const response = await fetch(commitsApiUrl, {
            method: 'GET',
            headers: {
                'PRIVATE-TOKEN': token
            }
        });

        if (!response.ok) {
            throw new Error(`GitLab API error: ${response.status} - ${response.statusText}`);
        }

        let commits = await response.json();

        commits = commits.slice(0, 2)

        displayCommits(commits);

    } catch (error) {
        handleError(error.message);

    } finally {
        // Remove spinner
        loadBranchesBtn.classList.remove('loading');
        loadBranchesBtn.disabled = false;
    }
}


// Function to display the commits in the table
async function displayCommits(commits) {
    const table = document.getElementById('branchTable');
    const tableBody = document.getElementById('branchTableBody');
    table.style.display = 'table';
    tableBody.innerHTML = '';

    // Fetch the current installed version
    await fetchVersion();

    // Only display the latest 2 commits
    if (commits.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4">No commits found.</td></tr>';
    } else {
        commits.forEach((commit, index) => {
            let row = document.createElement('tr');

            const installCell = commit.message.trim() === currentVersion.trim()
                ? `<td style="color: green; font-weight: bold;">Installed</td>`
                : `<td style="cursor: pointer;text-decoration: underline; color: blue;"
                onclick="install('API-Server-ML',
                '${commit.message.replace(/'/g, "\\'").replace(/\n/g, "")}')">Install</td>`;

            row.innerHTML = `
                <td>API-Server-ML</td>
                <td>${commit.message}</td>
                <td>${new Date(commit.authored_date).toLocaleString()}</td>
                ${installCell}
            `;

            tableBody.appendChild(row);
        });
    }

}


function handleError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorMessage.style.color = "red";
    errorMessage.style.fontWeight = "bold";
    console.error(message);
}

document.getElementById('loadBranchesBtn').addEventListener('click', fetchCommits);

// New module
// Fetch cameras and populate the table
async function fetchCameras() {
    const cameraContainer = document.getElementById("edit-camera-container");
    cameraContainer.style.display = "block";
    try {
        const response = await fetch('/cameras');
        if (!response.ok) {
            throw new Error('Failed to fetch cameras');
        }
        const cameras = await response.json();
        console.log('Fetched cameras:', cameras); // Debugging: Check fetched data
        const tableBody = document.getElementById("table_body");
        if (!tableBody) {
            console.error('Table body not found');
            return;
        }

        // Clear existing rows
        tableBody.innerHTML = '';

        // Loop through the camera data and create table rows
        cameras.forEach((camera) => {
            const row = document.createElement('tr');
            
            const activeText = document.createElement("span");
            activeText.innerText = "Active";
            activeText.style.color = "#336600";
            activeText.style.fontWeight = 700;

            const inactiveText = document.createElement("span");
            inactiveText.innerText = "Inactive";
            inactiveText.style.color = "#993300";
            inactiveText.style.fontWeight = 700;

            // Add camera data to the row
            row.innerHTML = `
                <td>${camera.camera_id}</td>
                <td>${camera.camera_url}</td>
                <td>${camera.camera_type}</td>
                <td></td>
                <td></td>
                <td></td>
            `;

            row.children[3].appendChild(camera.camera_running_status ? activeText : inactiveText);

            const editButton = document.createElement("button");
            editButton.innerText = "Edit";
            editButton.style.backgroundColor = "#FFFF00";
            editButton.addEventListener("click", () => openEditForm(camera));
            row.children[4].appendChild(editButton);

            // Delete button
            const deleteButton = document.createElement("button");
            deleteButton.innerText = "Delete";
            deleteButton.style.backgroundColor = "#FF0000";
            deleteButton.addEventListener("click", () => deleteCamera(camera.camera_id));
            row.children[5].appendChild(deleteButton);

            // Append the row to the table body
            tableBody.appendChild(row);
        });

        console.log('Table populated successfully'); // Debugging: Confirm rows are appended
    } catch (error) {
        console.error('Error fetching cameras:', error);
        alert('Failed to fetch cameras. Check the console for details.');
    }
}

// Function to delete camera
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
            await fetchCameras();
            setTimeout(() => location.reload(), 1000);
        } catch(error) {
            console.error("Error deleting camera: ", error);
            alert("Failed to delete camera. Check the console for details");
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
    document.getElementById("open_edit_form").style.display = "block";
}

async function editCamera(event) {
    // Prevent the default form submission behavior
    event.preventDefault();

    // Get the camera_id from the input field
    const camera_id = parseInt(document.getElementById("camera_id").value);
    const camera_url = document.getElementById("camera_url").value;
    const camera_type = document.getElementById("camera_type").value;
    const camera_running_status = document.getElementById("camera_running_status").checked; // Use `.checked` for checkbox
    const threshold = document.getElementById("threshold").value;
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

        // Call a function to refresh the cameras list
        await fetchCameras();

        // Reload the page after 1 second to reflect the changes
        setTimeout(() => location.reload(), 1000);
    } catch (error) {
        console.error("Error editing camera: ", error);
        alert("Error editing camera. Check the console for details.");
    }
}
// Fetch cameras when user clicks a button
document.getElementById("edit_camera_stats").addEventListener("click", fetchCameras);
// Example of calling openEditForm when an edit button is clicked
document.getElementById("edit-form").addEventListener("submit", editCamera);


