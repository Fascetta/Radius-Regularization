#!/bin/bash

# war stands for "wait and run". This script waits for a PID to finish and then runs a command.

# Example:
# bash wait_and_run.sh configs/gtav/source_target.yaml 28281
pid_to_wait_for="$1"
command_to_run="bash train.sh"

# Wait for the PID to finish
echo "Waiting for PID $pid_to_wait_for to finish..."
while ps -p $pid_to_wait_for > /dev/null
do
    echo "Still waiting for $pid_to_wait_for"
    sleep 60
done

# Run the command
echo "PID $pid_to_wait_for has finished. Running '$command_to_run'..."
eval "$command_to_run"