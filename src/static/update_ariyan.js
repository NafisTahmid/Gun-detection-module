const token = 'y_1htLH2vs3My9CtGXk5';
const fixedBranchName = 'API-Server-ML'; // Fixed branch name
const repoApiUrl = 'https://gitlab.accelx.net/api/v4/projects/152/repository/commits?ref_name=develop';
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
    } catch (error) {
        console.error("Failed to fetch version:", error.message);
    }
}

// Function to fetch commits for the fixed branch
async function fetchCommits() {
    try {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = "";
        const loadCommitsBtn = document.getElementById('loadCommitsBtn');
        loadCommitsBtn.classList.add('loading');
        loadCommitsBtn.disabled = true;

        const response = await fetch(repoApiUrl, {
            method: 'GET',
            headers: {
                'PRIVATE-TOKEN': token
            }
        });

        if (!response.ok) {
            throw new Error(`GitLab API error: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        displayCommits(data);
    } catch (error) {
        handleError(error.message);
    } finally {
        const loadCommitsBtn = document.getElementById('loadCommitsBtn');
        loadCommitsBtn.classList.remove('loading');
        loadCommitsBtn.disabled = false;
    }
}

// Function to display commits in a table
async function displayCommits(commits) {
    const table = document.getElementById('commitTable');
    const tableBody = document.getElementById('commitTableBody');
    table.style.display = 'table';
    tableBody.innerHTML = '';

    await fetchVersion();

    if (commits.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3">No commits found for this branch.</td></tr>';
    } else {
        commits.slice(0, 3).forEach(commit => { // Limit to last 3 commits
            const row = document.createElement('tr');

            const installCell = commit.message === currentVersion
                ? `<td style="color: green; font-weight: bold;">Installed</td>`
                : `<td style="cursor: pointer;text-decoration: underline; color: blue;" onclick="install('${commit.id}', '${commit.message}')">Install</td>`;

            row.innerHTML = `
                <td>${fixedBranchName}</td> <!-- Fixed branch name -->
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

document.getElementById('loadBranchesBtn').addEventListener('click', fetchBranches);



