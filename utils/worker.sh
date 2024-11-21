#!/bin/bash

# .env 파일 경로
ENV_FILE="/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/.env"

# .env 파일 로드
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo ".env 파일을 찾을 수 없습니다: $ENV_FILE"
    exit 1
fi

# ROOT_DIR 기반으로 경로 설정
QUEUE_FILE="$ROOT_DIR/job_queue.txt"
LOG_FILE="$ROOT_DIR/job_worker.log"

source ~/.bashrc

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        # 첫 번째 줄 읽기
        JOB=$(head -n 1 "$QUEUE_FILE")

        # 현재 시간 기록
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        # 유효한 명령어인지 확인
        if [[ ! "$JOB" =~ ^[a-zA-Z0-9_./-]+ ]]; then
            echo "[$TIMESTAMP] 잘못된 명령어: $JOB" >> "$LOG_FILE"
            # 잘못된 명령어 제거
            sed -i '1d' "$QUEUE_FILE"
            continue
        fi
        # 명령어 실행 및 로그 기록
        echo "[$TIMESTAMP] 실행 중: $JOB" >> "$LOG_FILE"

        if eval "$JOB" >> "$LOG_FILE" 2>&1; then
            # 작업이 성공적으로 완료된 경우
            echo "[$TIMESTAMP] 완료: $JOB" >> "$LOG_FILE"
        else
            # 작업에 에러가 발생한 경우
            echo "[$TIMESTAMP] 에러 발생: $JOB" >> "$LOG_FILE"
        fi

        # 첫 번째 줄 삭제
        sed -i '1d' "$QUEUE_FILE"
    fi
    # 짧은 대기 후 다시 확인
    sleep 2
done
