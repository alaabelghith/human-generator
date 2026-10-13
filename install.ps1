# Human Generator installer (Windows, NVIDIA GPU).
# Run from the repository folder:   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   -HoudiniVersion 20.5   which Houdini preferences folder gets the package (default: every houdiniXX.X folder found)
#   -Cuda cu128            PyTorch CUDA build (cu128 covers RTX 20xx to 50xx)
#   -SkipPackage           do not write the Houdini package file
param(
  [string]$HoudiniVersion = "",
  [string]$Cuda = "cu128",
  [switch]$SkipPackage
)
# "Continue": Windows PowerShell 5 turns a native tool's progress output on stderr into a terminating error
$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

# Pinned versions: the combination this release was tested with
$TorchVersion = "2.11.0"
$Kimodo = "kimodo @ git+https://github.com/nv-tlabs/kimodo@58e781898b3d7e328a676a75d3e338c45dce3ad9"
$SomaX = "py-soma-x[anny]==0.3.3"
$Anny = "anny==0.2.0"

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit code $LASTEXITCODE)" -ForegroundColor Red; exit 1 } }

Step "Checking uv (Python package manager)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv not found: installing it from astral.sh"
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version; Check "uv"

Step "Creating the Python 3.10 environment in .venv"
if (-not (Test-Path $Py)) { uv venv --python 3.10 $Venv; Check "creating .venv" }

Step "Installing PyTorch $TorchVersion ($Cuda)"
uv pip install --python $Py "torch==$TorchVersion" --index-url "https://download.pytorch.org/whl/$Cuda"; Check "installing PyTorch"

Step "Installing Kimodo, SOMA-X and Anny"
# skip Kimodo's optional C++ foot-contact post-process (needs CMake + a compiler); Human Generator does not use it
$env:SKIP_MOTION_CORRECTION_IN_SETUP = "1"
uv pip install --python $Py $Kimodo $SomaX $Anny "huggingface_hub"; Check "installing Kimodo / SOMA-X / Anny"

Step "Checking the GPU"
& $Py -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else '(no GPU: the server will run on the CPU, very slowly)')"

if (-not $SkipPackage) {
  Step "Installing the Houdini package"
  $prefs = Join-Path ([Environment]::GetFolderPath("MyDocuments")) ""
  $folders = if ($HoudiniVersion) { @(Join-Path $prefs "houdini$HoudiniVersion") } else { Get-ChildItem $prefs -Directory -Filter "houdini*" | ForEach-Object FullName }
  if (-not $folders) { Write-Host "No Documents\houdiniXX.X folder found: start Houdini once, then run install.ps1 again." -ForegroundColor Yellow }
  $rootFwd = $Root -replace "\\", "/"
  $pkg = @{ env = @(@{ HUMAN_GENERATOR = $rootFwd }); hpath = "`$HUMAN_GENERATOR/houdini" } | ConvertTo-Json -Depth 4
  foreach ($f in $folders) {
    $dir = Join-Path $f "packages"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Set-Content -Path (Join-Path $dir "human_generator.json") -Value $pkg -Encoding ASCII
    Write-Host "  wrote $dir\human_generator.json"
  }
}

Step "Hugging Face access (needed once, to download the models)"
$who = & $Py -c "from huggingface_hub import whoami`ntry: print(whoami()['name'])`nexcept Exception: pass"
if ($LASTEXITCODE -ne 0 -or -not $who) {
  Write-Host @"
Kimodo's text encoder is Meta's Llama 3 (gated). Before the first motion:
  1. Ask for access:  https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  2. Make a READ token: https://huggingface.co/settings/tokens
  3. Log in (paste the token when asked, never into a chat):
       .\.venv\Scripts\hf.exe auth login
"@ -ForegroundColor Yellow
} else {
  Write-Host "  logged in as $who"
}

Step "Done"
Write-Host @"
Restart Houdini, then in a Geometry network: Tab > Human Generator.
The first cook starts the server (server\start_server.bat) and downloads the body models (~1 GB);
the first Generate Motion downloads Kimodo and Llama 3 (~17 GB). Later starts run offline.
"@
