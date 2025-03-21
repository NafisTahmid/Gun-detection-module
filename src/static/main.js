

function showForm() {
    //document.getElementById('change_server_warning').style.display = 'none';
    document.getElementById('button-cluster').style.display = 'none';
    document.getElementById('change_server_form').style.display = 'block';

    showToast('Warning: Changing the video server ID will result in the permanent loss of all data. Please consult with the administrator before proceeding with any modifications.');
}

function hideForm() {
    //document.getElementById('change_server_warning').style.display = 'block';
    document.getElementById('button-cluster').style.display = 'block';
    document.getElementById('change_server_form').style.display = 'none';

    const toastContainer = document.getElementById('toast-container');
    while (toastContainer.firstChild) {
        toastContainer.removeChild(toastContainer.firstChild);
    }
}

async function install(branch, version) {
    const loadingOverlay = document.getElementById('loading_overlay');
    loadingOverlay.style.display = 'flex'; // Show loading overlay
    loadingOverlay.style.opacity = '1'; // Make it fully visible

    try {
        console.log('Installing branch:', branch, 'Version:', version);
        // Perform POST request to install the selected branch
        const response = await fetch('/install_update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',  // This header is essential
            },
            body: JSON.stringify({
                branch: branch,
                version: version // (Nabil)
            })
        });

        // Parse the response data
        const data = await response.json();
        console.log(data);

        // Handle successful response
        if (data.success) {
            // Update UI or display success message
            document.getElementById('install_status').innerText = `Successfully installed ${branch} branch.`;
            document.getElementById('install_status').style.color = "green";
            document.getElementById('install_status').style.fontWeight = 'bold';
        } else {
            document.getElementById('loading_message').innerText = `Installing... Please Wait`;
            document.getElementById('install_status').innerText = `Processing installation of ${branch} branch.`;
            document.getElementById('install_status').style.color = "blue";
            document.getElementById('install_status').style.fontWeight = 'bold';
        }

    } catch (error) {
        console.error('Error installing branch:', error);
        document.getElementById('install_status').innerText = `Error installing ${branch} branch.`;
        document.getElementById('install_status').style.color = "red";
        document.getElementById('install_status').style.fontWeight = 'bold';
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
            loadingOverlay.style.opacity = '0';
        }, 1000);
    } finally {
        document.getElementById('loading_message').innerText = `Processing... Please Wait`;
        setTimeout(() => {
            const checkServerStatus = async () => {
                try {
                    const statusResponse = await fetch('/check_server', {
                        method: 'GET',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    return statusResponse.message !== "No Server Running" && statusResponse.status !== 500;
                } catch {
                    return false; // Server is still down
                }
            };

            const waitForServer = async () => {
                document.getElementById('loading_message').innerText = `Restarting... Please Wait`;
                let serverOnline = false;
                while (!serverOnline) {
                    serverOnline = await checkServerStatus();
                    if (!serverOnline) {
                        await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds before retry
                    }
                }
                window.location.reload(true);
            };

            waitForServer();
        },


            1000)

    }
}



async function submitForm(event) {
    event.preventDefault();
    const videoServerId = document.getElementById('video_server_id').value;
    document.getElementById('video_server_id').value = ''
    const response = await fetch('/', {
        method: 'POST',
        body: new URLSearchParams({
            video_server_id: videoServerId
        })
    });
    const data = await response.json();
    handleRestartClick(1)
    if (data.video_server_status) {

        document.getElementById('video_server_status').innerText = data.video_server_status;
        document.getElementById('video_server_status').style.color = "#172554";
        document.getElementById('video_server_status').style.fontWeight = 'bold';

    }
    document.getElementById('change_server_form').style.display = 'none';
    document.getElementById('change_server_warning').style.display = 'block';
    document.getElementById('change_server_button').style.display = 'block';
    document.getElementById('button-cluster').style.display = 'block';
}



function handleRestartClick(factory_reset = 0) {
    const loadingOverlay = document.getElementById('loading_overlay');
    loadingOverlay.style.display = 'flex'; // Show loading overlay
    loadingOverlay.style.opacity = '1'; // Make it fully visible

    (async function () {
        try {
            const request_data = { trigger: factory_reset }
            const response = await fetch('/restart_server', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request_data)
            });
            const result = await response.json();

            if (result.status === 'success') {
                // Polling the server to check if it's back online
                const checkServerStatus = async () => {
                    try {
                        const statusResponse = await fetch('/check_server', {
                            method: 'GET',
                            headers: { 'Content-Type': 'application/json' }
                        });
                        return statusResponse.message !== "No Server Running" && statusResponse.status !== 500;
                    } catch {
                        return false; // Server is still down
                    }
                    return false;
                };

                const waitForServer = async () => {
                    let serverOnline = false;
                    while (!serverOnline) {
                        serverOnline = await checkServerStatus();
                        if (!serverOnline) {
                            await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds before retry
                        }
                    }
                    window.location.reload(); // Reload the page once the server is back online
                };

                waitForServer();
            } else {
                alert('Failed to restart server: ' + result.message);
                loadingOverlay.style.display = 'none'; // Hide loading overlay if restart fails
                loadingOverlay.style.opacity = '0';
            }
        } catch (error) {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none'; // Hide loading overlay on error
            loadingOverlay.style.opacity = '0';
        }
    })();
}