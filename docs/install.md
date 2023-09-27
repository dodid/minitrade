# Installation

## Hardware requirements

Minitrade intends to run on very low cost machines such as AWS Lightsail instances with 1 GB RAM, 1 vCPU, 40 GB SSD, that cost $5 per month. It's recommended to add 1GB swap space following this [guidence](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04) or similar. 

## Security recommendations

Securing server access is crucial due to locally saved broker credentials. Here are some security recommendations:

- Use a dedicated server solely for Minitrade, avoiding mixed-purpose instances.
- Install OpenVPN for remote server access.
- Set up firewall rules to allow only SSH and OpenVPN connections.
- Use firewall rules to restrict access to the Minitrade web UI and IB gateway by blocking all ports except SSH and OpenVPN. Access the web UI via a private IP after connecting with OpenVPN.
- [Create a dedicated IB user](https://www.interactivebrokers.com/en/software/singlefunds/topics/fundsaddusers.htm) for Minitrade, granting only necessary trading access and disallowing other privileges to avoid conflicts and minimize account permissions.

## Try in docker

Although running Minitrade in a Docker container is not recommended due to resource consumption and management complexity, it is possible for experimentation purposes. If you choose to do so, follow these steps:

1. Build a Docker image from the provided [Dockerfile](https://github.com/dodid/minitrade/blob/main/Dockerfile).
2. Expose port 8501 to access the web UI.
3. Note that the image is only tested on Linux hosts and may not work on Mac with M1/M2 chips due to compatibility issues with Chrome.
4. Restart the container if you modify the telegram or email settings to make them effective.

## Install on Ubuntu 20.04 

1. Install OpenVPN, following the instructions [here](https://www.cyberciti.biz/faq/ubuntu-20-04-lts-set-up-openvpn-server-in-5-minutes/). Make sure firewall is open for the port that OpenVPN listens on.

2. Install pyenv, following the instructions [here](https://brain2life.hashnode.dev/how-to-install-pyenv-python-version-manager-on-ubuntu-2004). Then install python 3.10:
   
        pyenv install 3.10.10
        pyenv global 3.10.10

3. Install dependencies

        sudo apt install -y default-jre
        
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
        sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
        sudo apt update
        sudo apt install -y google-chrome-stable

4. Install Minitrade

        pip install minitrade
        minitrade init


## Install on Ubuntu 22.04

1. Install OpenVPN, following the instructions [here](https://www.cyberciti.biz/faq/ubuntu-22-04-lts-set-up-openvpn-server-in-5-minutes/). Make sure firewall is open for the port that OpenVPN listens on.

2. Install python 3.10

        sudo apt update
        sudo apt install -y python3.10 wget gnupg
        sudo ln -s /usr/bin/python3.10 /usr/bin/python
        wget https://bootstrap.pypa.io/get-pip.py
        sudo python get-pip.py

3. Install dependencies

        sudo apt install -y default-jre
        
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
        sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
        sudo apt update
        sudo apt install -y google-chrome-stable

4. Install Minitrade

        pip install minitrade
        minitrade init


## Launch Minitrade

Launching Minitrade involves launching all three processes:

```
# start scheduler
minitrade scheduler start

# start ibgateway
minitrade ib start 

# start web UI
minitrade web start
```

You can use a process monitor like Supervisor for long-term process management. Alternatively, you can use [this script](https://github.com/dodid/minitrade/blob/main/mtctl.sh) for a quick and simple launch.
