FROM  python:3.12

RUN pip3 install git+https://github.com/gaarangoa/samv2.git
RUN apt-get install wget

RUN pip install matplotlib==3.9.1.post1
RUN pip install requests==2.32.3
RUN pip install pytest==8.3.2

RUN pip install jupyterlab==4.2.4

RUN pip install git+https://github.com/gaarangoa/samecode.git
RUN pip install seaborn
RUN pip install scipy==1.14.0
RUN pip install scikit-learn==1.5.1