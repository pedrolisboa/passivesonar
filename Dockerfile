FROM tensorflow/tensorflow:2.6.0rc1-gpu

RUN apt-get update
RUN apt-get install -y libsndfile1

RUN useradd -r -u 11659 -ms /bin/bash tars

ADD requirements.txt .

RUN pip install --upgrade setuptools==40.6.3
RUN pip install -r requirements.txt
RUN pip install jupyter

USER tars

WORKDIR /home/tars
