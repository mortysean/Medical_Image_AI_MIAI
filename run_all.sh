#!/bin/bash
# 一键启动 MIAI 前后端
# 用法: ./run_all.sh start | stop | restart | status

BACKEND_DIR="/home/seanhuang/MIAI/backend"
FRONTEND_DIR="/home/seanhuang/MIAI/frontend"
LOG_DIR="/home/seanhuang/MIAI/logs"

BACKEND_PORT=8080   # 后端 uvicorn 端口
FRONTEND_PORT=8010  # 前端静态端口

mkdir -p "$LOG_DIR"

start_backend() {
  echo ">>> 启动后端 (uvicorn, port=$BACKEND_PORT)"
  cd "$BACKEND_DIR"
  nohup uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT \
    > "$LOG_DIR/backend.log" 2>&1 &
  echo $! > "$LOG_DIR/backend.pid"
}

start_frontend() {
  echo ">>> 启动前端 (http.server, port=$FRONTEND_PORT)"
  cd "$FRONTEND_DIR"
  nohup python3 -m http.server $FRONTEND_PORT \
    > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$LOG_DIR/frontend.pid"
}

stop_backend() {
  if [ -f "$LOG_DIR/backend.pid" ]; then
    kill -9 $(cat "$LOG_DIR/backend.pid") 2>/dev/null
    rm -f "$LOG_DIR/backend.pid"
    echo ">>> 后端已停止"
  else
    echo ">>> 后端未运行"
  fi
}

stop_frontend() {
  if [ -f "$LOG_DIR/frontend.pid" ]; then
    kill -9 $(cat "$LOG_DIR/frontend.pid") 2>/dev/null
    rm -f "$LOG_DIR/frontend.pid"
    echo ">>> 前端已停止"
  else
    echo ">>> 前端未运行"
  fi
}

case "$1" in
  start)
    start_backend
    start_frontend
    ;;
  stop)
    stop_backend
    stop_frontend
    ;;
  restart)
    $0 stop
    sleep 2
    $0 start
    ;;
  status)
    echo "后端 PID: $(cat $LOG_DIR/backend.pid 2>/dev/null || echo not running)"
    echo "前端 PID: $(cat $LOG_DIR/frontend.pid 2>/dev/null || echo not running)"
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
