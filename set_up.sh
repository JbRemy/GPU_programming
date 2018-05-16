#!/usr/bin/env bash

N=${1:-'N'}

mkdir ./files

for i in {1..N} do
    mkdir ./files/$i