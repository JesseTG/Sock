#!/bin/bash
#PBS -keo
#PBS -qshort
#PBS -l nodes=1:ppn=28,walltime=4:00:00

module load shared
module load anaconda/.anaconda
module load rust

WORKING_DIR="$HOME/scratch/twitter"
DATASETS="$WORKING_DIR/datasets"
CAVERLEE_2011="$DATASETS/caverlee-2011"
CRESCI_2015="$DATASETS/cresci-2015"
CRESCI_2017="$DATASETS/cresci-2017"
NBC="$DATASETS/nbc"
FIVE38="$DATASETS/russian-troll-tweets"


cd "$WORKING_DIR"

function extract-caverlee-2011() {
    cat "$1/legitimate_users.txt" | cut -f1 > "$1/humans-caverlee-2011.ids"
    twarc users "$1/humans-caverlee-2011.ids" | jq -r .screen_name > "$1/humans-caverlee-2011.unsorted"
    sort -u "$1/humans-caverlee-2011.unsorted" > "$1/humans-caverlee-2011.txt"
    rm "$1/humans-caverlee-2011.unsorted" "$1/twarc.log"
}

function extract-cresci-2017() {
    xsv select screen_name "$1/users.csv" | tail -n+2 | sort -u > "$1/humans-cresci-2017.txt"
}

function extract-nbc() {
    xsv select screen_name "$1/users.csv" | tail -n+2 | sort -u > "$1/bots-nbc.txt"
}

function extract-538() {
    xsv select author "$1/IRAhandle_tweets_all.csv" | tail -n+2 | sort -u > "$1/bots-538.txt"
}

extract-caverlee-2011 "$CAVERLEE_2011" &
extract-cresci-2017 "$CRESCI_2017/genuine_accounts.csv" &
extract-nbc "$NBC" &
extract-538 "$FIVE38" &

wait

function sort-humans() {
    cat "$CAVERLEE_2011/humans-caverlee-2011.txt" "$CRESCI_2017/genuine_accounts.csv/humans-cresci-2017.txt" | \
        sort --unique --ignore-case > "$WORKING_DIR/humans.txt"
}

function sort-bots() {
    cat "$NBC/bots-nbc.txt" "$FIVE38/bots-538.txt" | sort --unique --ignore-case > "$WORKING_DIR/bots.txt"
}

sort-humans &
sort-bots &

wait