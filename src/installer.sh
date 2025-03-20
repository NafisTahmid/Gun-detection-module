#!/bin/bash

pyinstaller server.spec && \
sudo rm /home/nvidia/acceleye-detection-app/previous_version/server && \
sudo mv /home/nvidia/acceleye-detection-app/current_version/server /home/nvidia/acceleye-detection-app/previous_version && \
sudo mv /home/nvidia/gun-detection-module/src/dist/server /home/nvidia/acceleye-detection-app/current_version
