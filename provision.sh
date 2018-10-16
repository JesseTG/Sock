#!/usr/bin/env sh

set -e
# Quit in error if any of these commands fails

apt-get update
apt-get dist-upgrade --yes
apt-get install --no-install-recommends --yes gcc libc-dev
# Need gcc to compile some Python modules

pip3.7 install -v --no-cache-dir --compile --requirement requirements/prod.txt
# Install the Python modules

apt-get purge --yes binutils gcc manpages libc-dev-bin libsqlite* *-dev
apt-get autoremove --yes
apt-get clean --yes
apt-get autoclean --yes
# Now we don't need gcc

rm -rf /var/cache /var/log /var/backups
mkdir -p /var/cache /var/log /var/backups
# Clean the caches for good measure