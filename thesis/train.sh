#!/bin/bash
for i in {0..35} #29
do
        echo "Creating session $i"
        tmux new-session -d -s $i "source activate safety_wrappers; taskset --cpu-list $i python3.8 -m thesis.main --flag $i";
done

