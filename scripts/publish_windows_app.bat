@echo off
setlocal

echo Publishing GameSecretaryApp...
dotnet publish windows_app\GameSecretaryApp\GameSecretaryApp.csproj -c Release -o dist\GameSecretaryApp --self-contained false

echo Done. Run dist\GameSecretaryApp\GameSecretaryApp.exe
pause
