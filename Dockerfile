FROM flatironinstitute/triqs:2.2.1

WORKDIR /
USER root
RUN apt-get update && apt-get install python-skimage -y
RUN apt-get install jupyter-nbextension-jupyter-js-widgets -y
RUN apt-get install python-widgetsnbextension -y
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
USER $NB_USER
WORKDIR /home/$NB_USER
COPY data/RUN_ME.ipynb /home/$NB_USER/RUN_ME.ipynb
COPY data/visualization_function.py /home/$NB_USER/visualization_function.py
COPY data/TB_functions.py /home/$NB_USER/TB_functions.py
COPY data/RUN_ME.ipynb /home/$NB_USER/RUN_ME.ipynb
COPY data/LNO_ap3_863_W90/w2w_hr.dat /home/$NB_USER/LNO_ap3_863_W90/w2w_hr.dat
COPY data/LNO_ap3_863_W90/w2w.wout /home/$NB_USER/LNO_ap3_863_W90/w2w.wout

CMD ["jupyter","notebook","--ip","0.0.0.0"]
