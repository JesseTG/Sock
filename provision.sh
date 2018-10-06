#!/usr/bin/env sh

apt-get update
apt-get dist-upgrade -y
apt-get autoremove -y
apt-get clean -y
apt-get autoclean -y
pip3.7 install --no-cache-dir --compile --requirement requirements/prod.txt