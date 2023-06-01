FROM --platform=linux/amd64 python:3.10.11-slim-bullseye 

RUN apt update
RUN apt install -y default-jre
RUN apt install -y wget
RUN apt install -y gnupg
RUN apt install -y procps

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
RUN apt update
RUN apt install -y google-chrome-stable

RUN pip install minitrade -U
RUN minitrade init

WORKDIR /root

COPY restart.sh /root/restart.sh

CMD /root/restart.sh
EXPOSE 8501 6666 6667