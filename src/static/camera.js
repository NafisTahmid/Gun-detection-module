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
        const tableBody = document.querySelector('#camera_table tbody');
        const numberOfColumns = tableBody.querySelectorAll('tr:first-child td').length;

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