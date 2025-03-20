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

async function updateCameras() {
    const div = document.getElementById("camera-update");
    let ul = document.createElement("ul");

    try {
        const response = await fetch("././server_config.json", {
            method: "GET"
        })
        const json_data = await response.json()
        const cameras = json_data["cameras"];
        let li = document.createElement("li");

        cameras.forEach((camera, index) => {
            console.log(camera);
            li.innerHTML = `
                ${camera.camera_url} <button>Update</button> <button>Delete</button>
            `
            ul.appendChild(li);
        })
        div.appendChild(ul);
    } catch {
        throw new Error("Cameras not found") 
    }
}

document.getElementById('loadBranchesBtn').addEventListener('click', fetchCommits);
document.getElementById("edit_camera_stats").addEventListener('click', updateCameras);



