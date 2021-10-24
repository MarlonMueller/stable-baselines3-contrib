#!/bin/bash

for i in {0..127}
do
        # Distribute main calls to isolated cores
        tmux new-session -d -s $i "source activate safety_wrappers; taskset --cpu-list $i python3.8 -m main --flag $i";
done

