FROM julia:1.10.2

ENV USER pluto
ENV JULIA_NUM_THREADS 100


ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/src/.julia

RUN useradd -m -d ${USER_HOME_DIR} ${USER} && mkdir -p ${USER_HOME_DIR}/bin &&\
    mkdir -p ${JULIA_DEPOPT_PATH}/environments/v1.10/ 

COPY . ${USER_HOME_DIR}/src
COPY Manifest.toml ${JULIA_DEPOT_PATH}/environments/v1.10/Manifest.toml
COPY Project.toml ${JULIA_DEPOT_PATH}/environments/v1.10/Project.toml
COPY bin/* ${USER_HOME_DIR}/bin
WORKDIR ${USER_HOME_DIR}/src
    
RUN julia --project=. -e "import Pkg; Pkg.activate(); Pkg.instantiate(); Pkg.precompile();" &&\
    chown -R ${USER} ${USER_HOME_DIR} &&\
    chmod -R g=u ${USER_HOME_DIR}

RUN apt-get update && apt-get install -y vim

USER ${USER}

EXPOSE 1234

CMD [ "julia", "--project=/home/pluto/src/", "-e", "import Pluto; Pluto.run(host=\"0.0.0.0\", port=1234, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)"]
