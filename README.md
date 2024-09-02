# Gun Detection Module

Brief description or introduction of your project.
## Table of Contents

    Introduction
    Getting Started
        Prerequisites
        Installation
    Usage
    Contributing
    License

## Introduction

The Gun Detection Module, once set up will ready your server to host cameras and apply gun detection AI on the capured frames from the cameras.

## Getting Started
### Prerequisites

Have git & a debian linux distro installed.

### Installation

    Clone the repository:

`git clone https://gitlab.accelx.net/ariyan/gun-detection-module.git`

    Navigate to the project directory:

`cd gun-detection-module`

    Execute the setup script with sudo:

`sudo bash setup.sh`

## Usage

After setup.sh is done installing everything necessary to get the system up and running, you don't need to do anything. Everything is automated and will start working on it's own. The setup.sh, upon execution, will create two new shell scripts. toggle_service.sh & remove.sh

You can execute the toggle_service.sh to start or stop the service according to your choice.
`sudo bash toggle_service.sh`
You can execute the remove.sh to remove everything installed so far using setup.sh.
`sudo bash remove.sh`
## Contributing

Currently we are not taking any contributors
