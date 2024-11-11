#!/bin/bash

SESSION_NAME="job_manager"
QUEUE_FILE="/data/ephemeral/home/level2-nlp-datacentric-nlp-11/job_queue.txt"
ENQUEUE_SCRIPT="/data/ephemeral/home/level2-nlp-datacentric-nlp-11/utils/enqueue.sh"
WORKER_LOG="/data/ephemeral/home/level2-nlp-datacentric-nlp-11/job_worker.log"
source ~/.bashrc
# tmux 세션이 이미 존재하는지 확인
tmux has-session -t "$SESSION_NAME" 2>/dev/null

if [ $? != 0 ]; then
    # 새 tmux 세션 생성 (detached 상태)
    tmux new-session -d -s "$SESSION_NAME" -n "Main" "watch -n 1 cat $QUEUE_FILE"
    # tmux 세션에서 마우스 모드 활성화
    tmux set-option -t "$SESSION_NAME" -g mouse on
    # 창을 수직으로 분할하여 하단에 새로운 패널 생성 (Pane 1)
    tmux split-window -h -t "${SESSION_NAME}:Main" -p 50
    tmux split-window -h -t "${SESSION_NAME}:Main.1" -p 50

    # 하단 패널을 다시 수평으로 분할하여 Pane 1과 Pane 2 생성

    # Pane 1에 enqueue.sh 스크립트 실행
    tmux send-keys -t "${SESSION_NAME}:Main.1" "$ENQUEUE_SCRIPT" C-m

    # Pane 2에 worker.log 모니터링 명령어 실행
    tmux send-keys -t "${SESSION_NAME}:Main.2" "tail -F $WORKER_LOG" C-m

    # 원하는 레이아웃으로 설정 (main-vertical은 상단 패널과 하단 패널을 수평으로 배치)
    tmux select-layout -t "$SESSION_NAME":Main main-horizontal
    tmux resize-pane -t "${SESSION_NAME}:Main.0" -y 5
fi

# tmux 세션에 연결
tmux attach-session -t "$SESSION_NAME"
