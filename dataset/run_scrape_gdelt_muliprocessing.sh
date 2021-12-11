#!/bin/bash
total_num_processes=5


for ((i=0; i<total_num_processes; i++));
do
	nohup python3 -u scrape_gdelt.py --process_number=$i --total_num_processes=$total_num_processes &
done

# note: the command you should use to run this with slurm (the job manager used by Stanford's FarmShare
# is nohup srun ./run_scrape_gdelt_muliprocessing.sh --time=48:00:00 &
# if you don't set --time, then the max time given to your program is automatically set to two hours (which
# probably won't be long enough for the script to finish)