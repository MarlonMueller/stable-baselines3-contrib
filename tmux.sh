#!/bin/bash

for i in {0..127}
do
        # Distribute main calls to isolated cores
        tmux new-session -d -s $i "source activate safetyWrappers; taskset --cpu-list $i python3.8 main.py --flag $i";
done

