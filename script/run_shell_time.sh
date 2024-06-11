# 等待first_process_id进程运行结束再运行
first_process_id=$1

echo 

host_name=$(hostname)
echo "waiting for process $first_process_id on $host_name " 

current_time=$(date)
echo "Current time is: $current_time" > time.out

while ps -p $first_process_id >> time.out; do
    sleep 10
    current_time=$(date)
    echo "Current time is: $current_time, still waiting! " >> time.out
done

echo "Current time is: $current_time, start run on $host_name after process $first_process_id. " > time.out

# 要运行的代码
# ./run_shell_robustness.sh REDDIT-BINARY 0
./run_train_test.sh train 0 5