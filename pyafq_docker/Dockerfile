###############################################################################
# Dockerfile to build pyAFQ
###############################################################################

# Use python base image
FROM python:3.11

ARG COMMIT

RUN apt-get update && apt-get install --assume-yes cmake clang
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Install pyAFQ
RUN pip install --no-cache-dir git+https://github.com/tractometry/pyAFQ.git@${COMMIT}
RUN pip install fslpy
RUN pyAFQ download

ENTRYPOINT ["pyAFQ"]
