# Install on Ubuntu 20.04

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


# Install on Ubuntu 22.04

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