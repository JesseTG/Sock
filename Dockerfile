FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7

WORKDIR /dont-bother
COPY . /dont-bother

RUN apk update && apk add --no-cache nodejs
RUN pip3 install --no-cache-dir --requirement requirements/dev.txt
RUN npm install

EXPOSE 80 443 22

ENV FLASK_APP /dont-bother/autoapp.py
ENV FLASK_DEBUG 0

CMD ["flask", "run"]