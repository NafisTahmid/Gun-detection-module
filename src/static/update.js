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


// Fetch all cameras and populate the table
async function fetchCameras() {
    const response = await fetch('/cameras');
    const data = await response.json();
    const tableBody = document.querySelector('#camera_table tbody');
    tableBody.innerHTML = '';

    data.cameras.forEach(camera => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${camera.camera_id}</td>
            <td>${camera.camera_url}</td>
            <td>${camera.camera_type}</td>
            <td>${camera.camera_running_status ? 'Active' : 'Inactive'}</td>
            <td>
                <button onclick="openEditModal(${camera.camera_id})">Edit</button>
                <button onclick="deleteCamera(${camera.camera_id})">Delete</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Open the edit modal with camera details
async function openEditModal(cameraId) {
    const response = await fetch(`/cameras/${cameraId}`);
    const camera = await response.json();

    document.getElementById('editCameraId').value = camera.camera_id;
    document.getElementById('editCameraUrl').value = camera.camera_url;
    document.getElementById('editCameraType').value = camera.camera_type;
    document.getElementById('editCameraStatus').value = camera.camera_running_status;
    document.getElementById('editThreshold').value = camera.threshold;
    document.getElementById('editThirdParty').value = camera.third_party;

    document.getElementById('editModal').style.display = 'block';
}

// Close the edit modal
function closeEditModal() {
    document.getElementById('editModal').style.display = 'none';
}

// Handle form submission to update a camera
document.getElementById('editCameraForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    const cameraId = document.getElementById('editCameraId').value;
    const updatedData = {
        camera_url: document.getElementById('editCameraUrl').value,
        camera_type: document.getElementById('editCameraType').value,
        camera_running_status: document.getElementById('editCameraStatus').value === 'true',
        threshold: parseFloat(document.getElementById('editThreshold').value),
        third_party: document.getElementById('editThirdParty').value === 'true'
    };

    const response = await fetch(`/cameras/${cameraId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedData)
    });

    if (response.ok) {
        closeEditModal();
        fetchCameras(); // Refresh the table
    } else {
        alert('Failed to update camera.');
    }
});

// Delete a camera
async function deleteCamera(cameraId) {
    if (confirm('Are you sure you want to delete this camera?')) {
        const response = await fetch(`/cameras/${cameraId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            fetchCameras(); // Refresh the table
        } else {
            alert('Failed to delete camera.');
        }
    }
}

// Fetch cameras when the page loads
document.addEventListener('DOMContentLoaded', fetchCameras);


