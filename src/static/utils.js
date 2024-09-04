function showToast(message) {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerText = message;
    toastContainer.appendChild(toast);

    // Show toast with animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);

    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 10000); // Delay to match the transition time
    }, 10000);
}

function updateTime() {
    const now = new Date();
    let hours = now.getHours();
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');

    // Determine AM or PM
    const period = hours >= 12 ? 'PM' : 'AM';

    // Convert from 24-hour to 12-hour format
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'

    // Format hours with leading zero if necessary
    const hoursString = String(hours).padStart(2, '0');

    // Create time string in 12-hour format
    const timeString = `${hoursString}:${minutes}:${seconds} ${period}`;
    document.getElementById('current_time').innerText = timeString;
}

setInterval(updateTime, 1000);
window.onload = updateTime;