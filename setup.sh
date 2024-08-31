#!/bin/bash

# Check if the script is run with sudo
if [ "$(id -u)" != "0" ]; then
    echo "Permission denied. Please use 'sudo' to run this script ->  sudo sh setup.sh"
    exit 1
fi

script_loc=$(pwd)

# OpenCV Installation Command
sh scripts/cv2_install.sh

echo "Proceeding Forward with the Installation...."

cd $script_loc

# Installing all the dependencies
bash scripts/dependency_install.sh $script_loc/scripts/requirements.txt

echo "Dependencies are up to date.... Continuing Installation"

# Accelerating Hardware
sudo nvpmodel -m 0
sudo jetson_clocks


# Creating Service 

# Check if the video_process service is already running
if systemctl is-active --quiet videoprocess; then
    # If running, stop and disable the service
    systemctl stop videoprocess
    systemctl disable videoprocess
    echo "Existing service 'videoprocess' stopped and disabled."
fi

# Remove existing videoprocess.service if it exists
rm -f /etc/systemd/system/videoprocess.service

cat > /etc/systemd/system/videoprocess.service << EOF
[Unit]
Description=Video Process (Acceleye)
After=network.target

[Service]
WorkingDirectory=$script_loc
ExecStart=/usr/bin/python3 $script_loc/src/server.py

TimeoutStopSec=30

Restart=always
RestartSec=120

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable videoprocess
systemctl start videoprocess

echo "Adding Sudoers entry..."
echo "$USER ALL=NOPASSWD: /bin/systemctl stop videoprocess, /bin/systemctl start videoprocess" | sudo EDITOR='tee -a' visudo >/dev/null


if [ ! -f "toggle_service.sh" ]; then
    cat > toggle_service.sh << EOF
#!/bin/bash

if [ "$(id -u)" != "0" ]; then
    echo "Permission denied. Please use 'sudo' to run this script ->  sudo sh toggle_service.sh"
    exit 1
fi

if systemctl is-active --quiet videoprocess; then
    # If running, stop the service
    systemctl stop videoprocess
    echo "Service 'videoprocess' stopped."
else
    # If not running, start the service
    systemctl start videoprocess
    echo "Service 'videoprocess' started."
fi

EOF

fi



if [ ! -f "remove.sh" ]; then
    cat > remove.sh << EOF
#!/bin/bash

if [ "$(id -u)" != "0" ]; then
    echo "Permission denied. Please use 'sudo' to run this script ->  sudo sh remove.sh"
    exit 1
fi

if [ -f "toggle_service.sh" ]; then
    rm -f toggle_service.sh
fi

systemctl stop videoprocess
systemctl disable videoprocess

rm -f /etc/systemd/system/videoprocess.service

systemctl daemon-reload

echo "Service 'videoprocess' has been removed from the system."

if [ -f "src/server_config.json" ]; then
    rm -f src/server_config.json
fi



rm -f remove.sh


EOF

fi


echo "Setup has been succesfully completed"