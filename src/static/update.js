const token = 'eCZQp4QYQLee7_dzVodr';
const repoApiUrl = 'https://gitlab.accelx.net/api/v4/projects/152/repository/branches';
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

async function fetchBranches() {
    try {
        // Add spinner
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = "";
        loadBranchesBtn.classList.add('loading');
        loadBranchesBtn.disabled = true;

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
        data.sort((a, b) => new Date(b.commit.authored_date) - new Date(a.commit.authored_date));
        displayBranches(data);

    } catch (error) {
        handleError(error.message);

    } finally {
        // Remove spinner
        loadBranchesBtn.classList.remove('loading');
        loadBranchesBtn.disabled = false;
    }
}

async function displayBranches(branches) {
    const table = document.getElementById('branchTable');
    const tableBody = document.getElementById('branchTableBody');
    table.style.display = 'table';
    tableBody.innerHTML = '';
    await fetchVersion();



    // Filter out the master branch
    const filteredBranches = branches.filter(branch => branch.name !== 'master');

    if (filteredBranches.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3">No branches found.</td></tr>';
    } else {
        filteredBranches.forEach(branch => {
            const row = document.createElement('tr');

            const installCell = branch.commit.message === currentVersion
                ? `<td style="color: green; font-weight: bold;">Installed</td>`
                : `<td style="cursor: pointer;text-decoration: underline; color: blue;" onclick="install('${branch.name}', '${branch.commit.message}')">Install</td>`;

            row.innerHTML = `
                <td>${branch.name}</td>
                <td>${branch.commit.message}</td>
                <td>${new Date(branch.commit.authored_date).toLocaleString()}</td>
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



