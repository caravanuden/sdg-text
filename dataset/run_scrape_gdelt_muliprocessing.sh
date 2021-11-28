#!/bin/bash
total_num_processes=5


for ((i=0; i<total_num_processes; i++));
do
	nohup python3 scrape_gdelt.py --process_number=$i --total_num_processes=$total_num_processes &
done