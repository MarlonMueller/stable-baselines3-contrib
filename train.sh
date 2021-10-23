#!/bin/bash

# Distribute main calls to isolated cores.
for i in {0..127}
do
        #echo "Creating session $i"
        tmux new-session -d -s $i "source activate safety_wrappers; taskset --cpu-list $i python3.8 -m thesis.main --flag $i";
done

