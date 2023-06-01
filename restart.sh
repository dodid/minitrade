#!/bin/bash

# kill existing processes
pkill -f minitrade
[ -f nohup.out ] && mv nohup.out nohup.out.old

# reload minitrade processes
nohup minitrade scheduler start &
nohup minitrade ib start &
nohup minitrade web &

# check if run in docker
if [ -f /.dockerenv ]; then
    # Wait for any process to exit
    wait -n
    # Exit with status of process that exited first
    exit $?
fi