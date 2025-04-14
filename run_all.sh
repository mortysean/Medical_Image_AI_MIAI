#!/bin/bash

# === Configuration ===
DJANGO_PORT=8010
DASH_PORT=8050
DJANGO_DIR="backend"
DASH_DIR="frontend"

echo "📦 [1/3] Starting Django backend (port $DJANGO_PORT)..."
cd $DJANGO_DIR
nohup python manage.py runserver 127.0.0.1:$DJANGO_PORT > ../backend.log 2>&1 &
DJANGO_PID=$!
cd ..

echo "📦 [2/3] Starting Dash frontend (port $DASH_PORT)..."
cd $DASH_DIR
nohup python dash_app.py > ../dash.log 2>&1 &
DASH_PID=$!
cd ..

echo "✅ [3/3] All services started!"
echo "🔁 Django PID: $DJANGO_PID | Log: backend.log"
echo "🌐 Dash   PID: $DASH_PID | Log: dash.log"
echo ""
echo "📍 Access URLs:"
echo "🔗 Django API: http://127.0.0.1:$DJANGO_PORT/api/predict/"
echo "🔗 Dash UI:   http://127.0.0.1:$DASH_PORT"
