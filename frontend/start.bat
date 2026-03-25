@echo off
SETLOCAL
cd /d %~dp0
npm install
set PORT=3001
npm start
