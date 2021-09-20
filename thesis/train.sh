#!/bin/bash
for i in {0..27}; tmux new-session -d -s $i 'taskset --cpu-list $i python3.8 -m thesis.main --flag $i'; done