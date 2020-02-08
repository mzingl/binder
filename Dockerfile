FROM flatironinstitute/triqs:latest

WORKDIR /
USER root
RUN apt-get update && apt-get install python-skimage -y
RUN apt-get install jupyter-nbextension-jupyter-js-widgets -y
RUN apt-get install python-widgetsnbextension -y
RUN apt-get update && sudo apt-get install -y software-properties-common apt-transport-https curl
RUN curl -L https://users.flatironinstitute.org/~ccq/triqs/bionic/public.gpg | sudo apt-key add -
RUN add-apt-repository "deb https://users.flatironinstitute.org/~ccq/triqs/bionic/ /"
RUN apt-get update
RUN git clone https://github.com/TRIQS/tprf tprf.src
RUN mkdir tprf.build
WORKDIR tprf.build
RUN CXX=/usr/bin/g++ cmake ../tprf.src
RUN make install
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
ARG NB_USER=triqs
ARG NB_UID=1000
RUN usermod -u $NB_UID -l $NB_USER -d /home/$NB_USER -m triqs
USER $NB_USER
WORKDIR /home/$NB_USER
COPY data/RUN_ME.ipynb /home/$NB_USER/RUN_ME.ipynb
COPY data/visualization_function.py /home/$NB_USER/visualization_function.py
COPY data/TB_functions.py /home/$NB_USER/TB_functions.py
COPY data/RUN_ME.ipynb /home/$NB_USER/RUN_ME.ipynb
COPY data/LNO_ap3_863_W90/w2w_hr.dat /home/$NB_USER/LNO_ap3_863_W90/w2w_hr.dat
COPY data/LNO_ap3_863_W90/w2w.wout /home/$NB_USER/LNO_ap3_863_W90/w2w.wout

CMD ["jupyter","notebook","--ip","0.0.0.0"]
