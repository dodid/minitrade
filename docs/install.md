# Installation

## Hardware requirements

Minitrade is designed to operate efficiently on resource-constrained environments, such as AWS Lightsail instances with modest specifications: 1 GB RAM, 1 vCPU, and 40 GB SSD, which typically cost $5 per month. To optimize performance on such machines, it is advisable to allocate an additional 1 GB of swap space. Detailed instructions for adding swap space on Ubuntu 20.04, or similar operating systems, can be found in this [guide](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04).

## Security recommendations

Ensuring server access security is paramount, especially considering the sensitive broker credentials stored locally. Here are key security recommendations:

- **Dedicated Server**: Employ a dedicated server exclusively for Minitrade operations to mitigate risks associated with mixed-purpose instances.
- **Secure Remote Access**: Use OpenVPN or similar technology for secure remote server access, safeguarding against unauthorized entry.
- **Firewall Configuration**: Configure firewall rules to permit only SSH and OpenVPN connections. Utilize firewall rules to block remote access to the Minitrade web UI and IB gateway, only allow local access to the web UI via a private IP address over OpenVPN connection.
- **Dedicated IB User**: Establish a [dedicated IB user](https://www.interactivebrokers.com/en/software/singlefunds/topics/fundsaddusers.htm) for Minitrade. This user should be granted only essential trading access while disallowing other privileges to prevent conflicts and minimize account permissions.

## Try in docker

While running Minitrade within a Docker container is generally discouraged due to resource utilization and management complexities, it can serve as a viable option for experimentation. If you opt to proceed with this approach, adhere to the following steps:

1. **Build Docker Image**: Begin by constructing a Docker image using the provided [Dockerfile](https://github.com/dodid/minitrade/blob/main/Dockerfile).
2. **Port Exposure**: Ensure port 8501 is exposed to facilitate access to the web UI.
3. **Platform Compatibility**: Note that the Docker image has been primarily tested on Linux hosts. It may encounter compatibility issues on Mac systems equipped with M1/M2 chips, particularly due to Chrome compatibility concerns.
4. **Configuration Modifications**: Should modifications be made to the telegram or email settings, it's imperative to restart the container to enact these changes effectively.


## Install on Ubuntu 20.04 

1. **Install OpenVPN**: Follow the instructions provided [here](https://www.cyberciti.biz/faq/ubuntu-20-04-lts-set-up-openvpn-server-in-5-minutes/) to set up OpenVPN. Ensure that the firewall allows traffic on the port OpenVPN listens on.

2. **Install pyenv**: Refer to the instructions outlined [here](https://brain2life.hashnode.dev/how-to-install-pyenv-python-version-manager-on-ubuntu-2004) to install pyenv. Once installed, proceed to install Python 3.10:

    ```bash
    pyenv install 3.11.8
    pyenv global 3.11.8
    ```

3. **Install Dependencies**:

    ```bash
    sudo apt install -y default-jre
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
    sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
    sudo apt update
    sudo apt install -y google-chrome-stable
    ```

4. **Install Minitrade**:

    ```bash
    pip install minitrade
    minitrade init
    ```


## Install on Ubuntu 22.04

1. **Install OpenVPN**: Follow the instructions provided [here](https://www.cyberciti.biz/faq/ubuntu-22-04-lts-set-up-openvpn-server-in-5-minutes/) to set up OpenVPN. Ensure that the firewall allows traffic on the port OpenVPN listens on.

2. **Install Python 3.10**:

    ```bash
    sudo apt update
    sudo apt install -y python3.10 wget gnupg
    sudo ln -s /usr/bin/python3.10 /usr/bin/python
    wget https://bootstrap.pypa.io/get-pip.py
    sudo python get-pip.py
    ```

3. **Install Dependencies**:

    ```bash
    sudo apt install -y default-jre
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
    sudo sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
    sudo apt update
    sudo apt install -y google-chrome-stable
    ```

4. **Install Minitrade**:

    ```bash
    pip install minitrade
    minitrade init
    ```

## Launch Minitrade

Launching Minitrade involves starting all three processes:

```bash
# Start the scheduler
minitrade scheduler start

# Start IB Gateway
minitrade ib start 

# Start the web UI
minitrade web start
```

For long-term process management, consider using a process monitor like Supervisor. Alternatively, you can use [this script](https://github.com/dodid/minitrade/blob/main/mtctl.sh) for a quick and simple launch.

