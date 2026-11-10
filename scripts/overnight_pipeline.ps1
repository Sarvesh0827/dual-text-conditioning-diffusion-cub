# Unattended post-VAE pipeline for the 4090 lab machine.
# Waits for train_vae to finish, then: eval_vae -> caches -> UNet C -> A -> B.
# Logs: runs/pipeline.log

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$DataRoot = "data\cub\CUB_200_2011"
$LogDir = Join-Path $Root "runs"
$Log = Join-Path $LogDir "pipeline.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $Log -Value $line
    Write-Host $line
}

function Test-VaeTrainingRunning {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*train_vae*' }
    return $null -ne $procs
}

function Get-CkptStep([string]$path) {
    if (-not (Test-Path $path)) { return -1 }
    $code = @"
import torch, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(int(ckpt.get('step', -1)))
"@
    $tmp = Join-Path $env:TEMP "read_ckpt_step.py"
    Set-Content -Path $tmp -Value $code -Encoding UTF8
    $out = & $Python $tmp $path 2>$null
    if ($LASTEXITCODE -ne 0) { return -1 }
    try { return [int]$out.Trim() } catch { return -1 }
}

function Invoke-Step([string]$name, [string[]]$argv) {
    Write-Log "START $name : $($argv -join ' ')"
    & $Python @argv 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Log "FAIL $name exit_code=$code"
        throw "Step failed: $name (exit $code)"
    }
    Write-Log "OK $name"
}

Write-Log "Overnight pipeline started in $Root"
Write-Log "Python=$Python"

# --- Wait for current VAE training to finish ---
Write-Log "Waiting for train_vae.py to exit..."
while (Test-VaeTrainingRunning) {
    Start-Sleep -Seconds 120
}
Write-Log "No train_vae process found."

$vaeLast = Join-Path $Root "runs\vae\vae_last.pt"
$vaeEma = Join-Path $Root "runs\vae\vae_ema.pt"
$step = Get-CkptStep $vaeLast
if ($step -lt 0) { $step = Get-CkptStep $vaeEma }
Write-Log "VAE checkpoint step=$step"

if ($step -lt 80000) {
    Write-Log "VAE not finished (step=$step < 80000). Attempting resume to completion..."
    Invoke-Step "train_vae_resume" @(
        "scripts\train_vae.py",
        "--config", "configs\vae.yaml",
        "--data_root", $DataRoot,
        "--out_dir", "runs\vae",
        "--resume", "auto"
    )
    $step = Get-CkptStep $vaeLast
    Write-Log "VAE step after resume pass=$step"
}

if (-not (Test-Path $vaeEma)) {
    Write-Log "ERROR: missing $vaeEma"
    exit 1
}
if ((Get-CkptStep $vaeEma) -lt 80000 -and (Get-CkptStep $vaeLast) -lt 80000) {
    Write-Log "ERROR: VAE still incomplete after resume attempt."
    exit 1
}

# Prefer ema weights for eval / downstream
$ckpt = $vaeEma

try {
    Invoke-Step "eval_vae" @(
        "scripts\eval_vae.py",
        "--config", "configs\vae.yaml",
        "--data_root", $DataRoot,
        "--ckpt", $ckpt,
        "--out_dir", "runs\vae"
    )

    Invoke-Step "cache_latents" @(
        "scripts\cache_latents.py",
        "--config", "configs\vae.yaml",
        "--data_root", $DataRoot,
        "--ckpt", $ckpt,
        "--out_dir", "cache\latents"
    )

    Invoke-Step "cache_clip" @(
        "scripts\cache_clip.py",
        "--config", "configs\default.yaml",
        "--data_root", $DataRoot,
        "--out_dir", "cache\clip"
    )

    foreach ($exp in @("C", "A", "B")) {
        $cfg = "configs\unet_$exp.yaml"
        $out = "runs\unet_$exp"
        Invoke-Step "train_unet_$exp" @(
            "scripts\train_unet.py",
            "--config", $cfg,
            "--data_root", $DataRoot,
            "--vae_ckpt", $ckpt,
            "--latent_cache", "cache\latents",
            "--clip_cache", "cache\clip",
            "--out_dir", $out,
            "--resume", "auto"
        )
    }

    Write-Log "PIPELINE COMPLETE"
} catch {
    Write-Log "PIPELINE STOPPED: $($_.Exception.Message)"
    exit 1
}
