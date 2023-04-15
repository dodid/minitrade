# Installation

## Hardware requirements

Minitrade intends to run on very low cost machines such as AWS Lightsail instances with 1 GB RAM, 1 vCPU, 40 GB SSD, that cost $5 per month. It's recommended to add 1GB swap space following this [guidence](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04) or similar. 

## Security recommendations

Securing the access to the server instance is super important since the broker credentials are saved locally. Try:

- Use a dedicated server for Minitrade and don't use the same instance for other purposes.
- Instal OpenVPN for remote access to the server
- Set up firewall rules to only allow SSH and OpenVPN

Minitrade web UI and IB gateway listen on all interfaces, therefore, accessible by public IP. It's important to use the firewall rules to block all ports but SSH and OpenVPN, and access the web UI via private IP after OpenVPN is connected.

[Create a dedicated IB user](https://www.interactivebrokers.com/en/software/singlefunds/topics/fundsaddusers.htm) for Minitrade. Only trading access should be provided, others, especially funding access, should be disallowed. This avoids access conflict when logging in from multiple devices and restricts the account permissions to the minimally required.

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
minitrade web
```

You can use any process monitor like Supervisor to keep the processes running. Or to do it quick and dirty, use a script like the following: 

```
#!/bin/bash

# kill existing processes
pkill -e -P 1 -f minitrade
[ -f nohup.out ] && mv nohup.out nohup.out.old

# reload processes
nohup minitrade scheduler start &
nohup minitrade ib start &
nohup minitrade web &
```