#!/bin/bash

##############################################################################
#  
#  healthcheck.sh
#
#  Bash script for checking a HTTP location
#
#  @author Wim Kosten <w.kosten@zeeland.nl>
#
##############################################################################

# Check for command line arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <host> <port> <path>"
    exit 1
fi

# set params
host=$1
port=$2
path=$3

# Check if netcat is installed
if ! command -v nc &> /dev/null; then
    # Netcat not installed
    exit 1
fi

# Send HTTP GET request using netcat
response=$(echo -e "GET $path HTTP/1.1\r\nHost: $host\r\nConnection: close\r\n\r\n" | nc $host $port)

# Check the HTTP response status code
if echo "$response" | head -n 1 | grep -q "HTTP/1.1 200"; then
    echo 0
else
    echo 1
fi

