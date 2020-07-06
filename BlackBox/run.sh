#!bin/bash
# Get Instance Type
JOB_NAME=$1

# Get profile result at half of 2epoch
sudo -i -u root bash << EOF
python3.6 /home/ubuntu/Deep-Cloud/job/test-profile.py --batch_size 16 --prof_start_batch 5625 --prof_end_batch 5626
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 16 --prof_start_batch 5625 --prof_end_batch 5626
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 32 --prof_start_batch 2812 --prof_end_batch 2813
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 64 --prof_start_batch 1405 --prof_end_batch 1406
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 128 --prof_start_batch 702 --prof_end_batch 703
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 256 --prof_start_batch 351 --prof_end_batch 352
sleep 3

python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 512 --prof_start_batch 175 --prof_end_batch 176
sleep 3

EOF
