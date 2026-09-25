<#
.SYNOPSIS
    Creates tools/parity/.venv, the reference Python stack of Gym.NET's live parity suite (tests/Gym.Tests.Parity).

.DESCRIPTION
    Builds a virtual environment next to this script from a CPython 3.12 interpreter and installs
    tools/parity/requirements.txt into it. The parity suite then finds the venv on its own: it asks the venv's interpreter
    for the base installation's python3XY.dll (a venv carries no DLL of its own, and pythonnet must load the base one) and
    for the venv's sys.path, so nothing has to be configured afterwards. GYMNET_PARITY_VENV points the suite at a venv
    elsewhere.

    Gymnasium is not installed. The suite imports it from the refs/Gymnasium submodule, so the oracle is exactly the
    pinned commit.

    Every run deletes and recreates the venv, so a drifted or half-installed venv is replaced, never patched.

.PARAMETER Python
    The interpreter to create the venv from. It must be CPython 3.12: pythonnet 3.0.5 drives 3.7 to 3.13, and 3.12 is the
    port plan's reference version, which tools/parity/manifest.json enforces at run time. Defaults to the Python
    launcher's 3.12 ("py -3.12") when the launcher is installed, otherwise to "python" on PATH.

.EXAMPLE
    pwsh tools/parity/setup_venv.ps1

.EXAMPLE
    pwsh tools/parity/setup_venv.ps1 -Python C:\Python312\python.exe
#>
[CmdletBinding()]
param(
    [string] $Python
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$venv = Join-Path $PSScriptRoot '.venv'
$requirements = Join-Path $PSScriptRoot 'requirements.txt'

# The interpreter is a command plus leading arguments, so that "py -3.12" is invoked the same way as a plain path.
if ($Python) {
    $interpreter = @($Python)
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $interpreter = @('py', '-3.12')
}
else {
    $interpreter = @('python')
}

<#
.SYNOPSIS
    Runs an executable and fails the script when it exits non-zero.
.DESCRIPTION
    Native commands do not throw on failure, even with $ErrorActionPreference = 'Stop', so every call goes through here;
    otherwise a failed pip install would be followed by a success message.
.PARAMETER Command
    The executable followed by its leading arguments, for example @('py', '-3.12').
.PARAMETER Arguments
    The arguments appended after the leading ones.
.OUTPUTS
    The command's standard output.
#>
function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string[]] $Command,
        [string[]] $Arguments = @()
    )
    $executable = $Command[0]
    $allArguments = @($Command | Select-Object -Skip 1) + $Arguments
    & $executable @allArguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$($Command -join ' ') $($Arguments -join ' ')' failed with exit code $LASTEXITCODE."
    }
}

# Refuse any other version up front: a venv built on it would install cleanly and then fail the manifest check
# (python.version 3.12.*) on every test run, far from the cause.
$version = (Invoke-Checked -Command $interpreter -Arguments @('-c', 'import sys; print("%d.%d" % sys.version_info[:2])') | Out-String).Trim()
if ($version -ne '3.12') {
    throw "'$($interpreter -join ' ')' is Python $version; the parity stack needs CPython 3.12. Pass -Python <path to python 3.12>."
}

if (Test-Path $venv) {
    Remove-Item -Recurse -Force $venv
}
Invoke-Checked -Command $interpreter -Arguments @('-m', 'venv', $venv)

$venvPython = Join-Path $venv 'Scripts\python.exe'
Invoke-Checked -Command @($venvPython) -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip')
Invoke-Checked -Command @($venvPython) -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirements)

# Import what the manifest pins, so a broken wheel fails here rather than in the first test.
Invoke-Checked -Command @($venvPython) -Arguments @('-c', 'import numpy, pygame, Box2D; print("numpy", numpy.__version__, "| pygame-ce", pygame.version.ver, "(SDL", str(pygame.version.SDL) + ") | Box2D", Box2D.__version__)')
Write-Host "Created $venv. The parity suite uses it automatically; see tests/Gym.Tests.Parity/CLAUDE.md."
