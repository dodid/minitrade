# Installation

## Hardware requirements

Minitrade intends to run on very low cost machines such as AWS Lightsail instances with 1 GB RAM, 1 vCPU, 40 GB SSD, that cost $5 per month.

## Security recommendations

Securing the access to the server instance is super important since the broker credentials are saved locally. Try:

- Use a dedicated server for Minitrade and don't use the same instance for other purposes.
- Instal OpenVPN for remote access to the server
- Set up firewall rules to only allow SSH and OpenVPN

Minitrade web UI and IB gateway listen on all interfaces, therefore, accessible by public IP. It's important to use the firewall rules to block all ports but SSH and OpenVPN, and access the web UI via internal IP after OpenVPN is connected.

[Create a dedicated IB user](https://www.interactivebrokers.com/en/software/singlefunds/topics/fundsaddusers.htm) for Minitrade. Only trading access should be provided, others, especially funding access, should be disallowed. This avoids access conflict when logging in from multiple devices and restricts the account permissions to the minimally required.

## Install on Ubuntu 20.04 

TODO: add instructions for setting up openvpn

1. Install pyenv and python 3.10

        sudo apt update

        sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        curl https://pyenv.run | bash

        exec $SHELL

        pyenv install 3.10.10

        pyenv global 3.10.10

        echo "export PATH=\"$HOME/.pyenv/bin:$PATH\"" >> .bashrc
        echo "eval \"$(pyenv init --path)\"" >> .bashrc
        echo "eval \"$(pyenv virtualenv-init -)\"" >> .bashrc

2. Install dependencies

        sudo apt install -y default-jre
        
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
        sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
        sudo apt update
        sudo apt install -y google-chrome-stable

3. Install Minitrade

        pip install minitrade
        minitrade init


## Install on Ubuntu 22.04

1. Install python 3.10

        sudo apt update
        sudo apt install -y python3.10 wget gnupg
        sudo ln -s /usr/bin/python3.10 /usr/bin/python
        wget https://bootstrap.pypa.io/get-pip.py
        sudo python get-pip.py

2. Install dependencies

        sudo apt install -y default-jre
        
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
        sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
        sudo apt update
        sudo apt install -y google-chrome-stable

3. Install Minitrade

        pip install minitrade
        minitrade init