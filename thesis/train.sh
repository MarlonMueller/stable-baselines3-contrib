#!/bin/bash
for i in {0..0}
do
        echo "Creating session $i"
        taskset --cpu-list $i tmux new-session -s $i "source activate safety_wrappers; python3.8 -m thesis.main --flag $i";
done