#!/bin/bash

ENV_FILE="/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/.env"

# .env 파일 로드
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo ".env 파일을 찾을 수 없습니다: $ENV_FILE"
    exit 1
fi

QUEUE_FILE="$ROOT_DIR/job_queue.txt"
clear
while true; do
    echo "추가할 작업 명령어를 입력하세요 (종료하려면 'exit' 입력):"
    read JOB_CMD

    if [ "$JOB_CMD" == "exit" ]; then
        echo "작업 추가를 종료합니다."
        break
    elif [ -n "$JOB_CMD" ]; then
        echo "$JOB_CMD" >> "$QUEUE_FILE"
        echo "작업이 job_queue.txt에 추가되었습니다: $JOB_CMD"
    else
        echo "명령어가 비어있습니다. 다시 입력해 주세요."
    fi
done
