FROM flatironinstitute/triqs:2.2.1

WORKDIR /
USER root
RUN apt-get update && apt-get install python-skimage -y
RUN apt-get install jupyter-nbextension-jupyter-js-widgets -y
RUN apt-get install python-widgetsnbextension -y
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
WORKDIR /home
COPY data/RUN_ME.ipynb /home/RUN_ME.ipynb
COPY data/visualization_function.py /home/visualization_function.py
COPY data/TB_functions.py /home/TB_functions.py
COPY data/RUN_ME.ipynb /home/RUN_ME.ipynb
COPY data/LNO_ap3_863_W90/w2w_hr.dat /home/LNO_ap3_863_W90/w2w_hr.dat
COPY data/LNO_ap3_863_W90/w2w.wout /home/LNO_ap3_863_W90/w2w.wout

CMD ["jupyter","notebook","--ip","0.0.0.0"]
