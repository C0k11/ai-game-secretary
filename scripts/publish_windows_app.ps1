param(
    [ValidateSet('Release','Debug')] [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "Publishing GameSecretaryApp -> dist/GameSecretaryApp ..." -ForegroundColor Cyan

dotnet publish "windows_app/GameSecretaryApp/GameSecretaryApp.csproj" -c $Configuration -o "dist/GameSecretaryApp" --self-contained false

Write-Host "Done. Run: dist/GameSecretaryApp/GameSecretaryApp.exe" -ForegroundColor Green
