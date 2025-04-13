#!/bin/bash

# === 配置 ===
DJANGO_PORT=8010
DASH_PORT=8050
DJANGO_DIR="backend"
DASH_DIR="frontend"

echo "📦 [1/3] 启动 Django 后端 (端口 $DJANGO_PORT)..."
cd $DJANGO_DIR
nohup python manage.py runserver 127.0.0.1:$DJANGO_PORT > ../backend.log 2>&1 &
DJANGO_PID=$!
cd ..

echo "📦 [2/3] 启动 Dash 前端 (端口 $DASH_PORT)..."
cd $DASH_DIR
nohup python dash_app.py > ../dash.log 2>&1 &
DASH_PID=$!
cd ..

echo "✅ [3/3] 全部启动完成！"
echo "🔁 Django PID: $DJANGO_PID | 日志: backend.log"
echo "🌐 Dash   PID: $DASH_PID | 日志: dash.log"
echo ""
echo "📍 访问地址："
echo "🔗 Django API： http://127.0.0.1:$DJANGO_PORT/api/predict/"
echo "🔗 Dash UI：   http://127.0.0.1:$DASH_PORT"
