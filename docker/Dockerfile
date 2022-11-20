FROM python:3.9
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./svm_gamma_0.001_C_0.5.joblib /exp/


RUN pip3 install -U scikit-learn
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
CMD ["python3","./serve.py"]
#build command: docker build -t exp:v1 -f docker/Dockerfile .
#run command: docker run -p 5000:5000 -it exp:v1