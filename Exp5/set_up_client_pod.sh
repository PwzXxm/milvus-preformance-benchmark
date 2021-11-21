#!/bin/bash

apt update
apt install python3-pip vim -y
pip3 install pymilvus pandas
yes | pip3 uninstall pymilvus
pip3 install -i https://test.pypi.org/simple/ pymilvus-perf==2.0.0rc9.dev5
