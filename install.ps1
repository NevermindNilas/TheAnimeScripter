& {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [string]$InstallPath,
        [switch]$AddToPath,
        [switch]$Force,
        [string]$Repository = "NevermindNilas/TheAnimeScripter"
    )

    $addToPathWasExplicit = $MyInvocation.BoundParameters.ContainsKey("AddToPath")

    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    function Write-Info {
        param([string]$Message)

        Write-Host "[TAS] $Message" -ForegroundColor Cyan
    }

    function Write-Success {
        param([string]$Message)

        Write-Host "[TAS] $Message" -ForegroundColor Green
    }

    function Get-InvokeWebRequestParams {
        param(
            [string]$Uri,
            [hashtable]$Headers,
            [string]$OutFile
        )

        $params = @{
            Uri     = $Uri
            Headers = $Headers
        }

        if ($OutFile) {
            $params.OutFile = $OutFile
        }

        if ((Get-Command Invoke-WebRequest).Parameters.ContainsKey("UseBasicParsing")) {
            $params.UseBasicParsing = $true
        }

        return $params
    }

    function Get-NormalizedPath {
        param([string]$Path)

        return [System.IO.Path]::GetFullPath($Path).TrimEnd("\\/")
    }

    function Test-PathEntryPresent {
        param(
            [string[]]$Entries,
            [string]$Candidate
        )

        $normalizedCandidate = Get-NormalizedPath -Path $Candidate
        foreach ($entry in $Entries) {
            if ([string]::IsNullOrWhiteSpace($entry)) {
                continue
            }

            try {
                if ((Get-NormalizedPath -Path $entry) -ieq $normalizedCandidate) {
                    return $true
                }
            }
            catch {
                if ($entry.TrimEnd("\\/") -ieq $normalizedCandidate) {
                    return $true
                }
            }
        }

        return $false
    }

    function Read-YesNoChoice {
        param(
            [string]$Prompt,
            [bool]$Default = $false
        )

        $defaultLabel = if ($Default) { "Y/n" } else { "y/N" }
        $response = Read-Host "$Prompt [$defaultLabel]"
        if ([string]::IsNullOrWhiteSpace($response)) {
            return $Default
        }

        switch -Regex ($response.Trim()) {
            '^(y|yes)$' { return $true }
            '^(n|no)$' { return $false }
            default {
                Write-Info "Please answer yes or no."
                return Read-YesNoChoice -Prompt $Prompt -Default $Default
            }
        }
    }

    function Resolve-AddToPathSelection {
        param(
            [bool]$WasExplicit,
            [bool]$RequestedValue
        )

        if ($WasExplicit) {
            return $RequestedValue
        }

        $envChoice = $env:TAS_INSTALL_ADD_TO_PATH
        if (-not [string]::IsNullOrWhiteSpace($envChoice)) {
            switch -Regex ($envChoice.Trim()) {
                '^(1|true|y|yes|on)$' { return $true }
                '^(0|false|n|no|off)$' { return $false }
                default {
                    Write-Info "Ignoring unrecognized TAS_INSTALL_ADD_TO_PATH value '$envChoice'."
                }
            }
        }

        try {
            return Read-YesNoChoice -Prompt "Add '$InstallPath' to your user PATH after installation?" -Default $false
        }
        catch {
            Write-Info "PATH will not be modified because no explicit choice was provided and prompting is unavailable."
            return $false
        }
    }

    if ($env:OS -ne "Windows_NT") {
        throw "install.ps1 only supports Windows."
    }

    $location = Get-Location
    if ($location.Provider.Name -ne "FileSystem") {
        throw "The current location must be a filesystem path."
    }

    if ([string]::IsNullOrWhiteSpace($InstallPath)) {
        $InstallPath = Join-Path $location.ProviderPath "TheAnimeScripter"
    }

    $InstallPath = Get-NormalizedPath -Path $InstallPath

    Write-Info "TheAnimeScripter will be installed to: $InstallPath"

    if (-not $PSCmdlet.ShouldProcess($InstallPath, "Install TheAnimeScripter portable CLI")) {
        return
    }

    $shouldAddToPath = Resolve-AddToPathSelection -WasExplicit $addToPathWasExplicit -RequestedValue $AddToPath.IsPresent
    if (-not $shouldAddToPath) {
        Write-Info "PATH will not be modified. Use -AddToPath if you want global commands without the prompt."
    }

    $managedPaths = @(
        "python.exe",
        "pythonw.exe",
        "python313.dll",
        "python313.zip",
        "main.py",
        "src",
        "Scripts",
        "tas-version-check.ps1",
        "TheAnimeScripter.cmd",
        "tas.cmd"
    )

    $existingManagedPaths = @(
        foreach ($managedPath in $managedPaths) {
            $candidate = Join-Path $InstallPath $managedPath
            if (Test-Path $candidate) {
                $candidate
            }
        }
    )

    if ($existingManagedPaths.Count -gt 0 -and -not $Force) {
        throw "Existing TheAnimeScripter files were detected in '$InstallPath'. Re-run with -Force to overwrite them."
    }

    $headers = @{
        "Accept"     = "application/vnd.github+json"
        "User-Agent" = "TheAnimeScripter-Install-Script"
    }

    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tas-install-" + [System.Guid]::NewGuid().ToString("N"))
    $downloadPath = $null
    $extractPath = $null

    try {
        Write-Info "Resolving the latest Windows release from GitHub..."
        $release = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$Repository/releases/latest" -Headers $headers
        $asset = @($release.assets | Where-Object { $_.name -match "^TAS-\d+-Windows\.zip$" }) | Select-Object -First 1

        if (-not $asset) {
            throw "No Windows release asset matching 'TAS-<version>-Windows.zip' was found in the latest release."
        }

        New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
        $downloadPath = Join-Path $temporaryRoot $asset.name
        $extractPath = Join-Path $temporaryRoot "expanded"

        Write-Info "Downloading $($asset.name)..."
        $downloadParams = Get-InvokeWebRequestParams -Uri $asset.browser_download_url -Headers $headers -OutFile $downloadPath
        Invoke-WebRequest @downloadParams

        Write-Info "Extracting the portable bundle..."
        Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force

        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null

        $expandedItems = @(Get-ChildItem -Path $extractPath -Force)
        if ($expandedItems.Count -eq 0) {
            throw "The downloaded archive did not contain any files to install."
        }

        foreach ($item in $expandedItems) {
            Copy-Item -Path $item.FullName -Destination $InstallPath -Recurse -Force
        }

        $versionCheckerContent = @"
param(
    [string]`$Repository = "NevermindNilas/TheAnimeScripter"
)

Set-StrictMode -Version Latest
`$ErrorActionPreference = "Stop"

`$installRoot = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$statePath = Join-Path `$installRoot "tas-version-state.json"
`$messagePath = Join-Path `$installRoot "tas-update-message.txt"
`$lockPath = Join-Path `$installRoot "tas-version-check.lock"
`$cooldown = [TimeSpan]::FromHours(12)

function Get-InstalledVersion {
    `$versionFile = Join-Path `$installRoot "src\version.py"
    if (-not (Test-Path `$versionFile)) {
        return `$null
    }

    `$match = [regex]::Match((Get-Content -Path `$versionFile -Raw), '__version__\s*=\s*"([^"]+)"')
    if (`$match.Success) {
        return `$match.Groups[1].Value
    }

    return `$null
}

function Remove-UpdateMessage {
    if (Test-Path `$messagePath) {
        Remove-Item -Path `$messagePath -Force -ErrorAction SilentlyContinue
    }
}

function Set-UpdateMessage {
    param(
        [string]`$InstalledVersion,
        [string]`$LatestTag,
        [string]`$ReleaseUrl
    )

    `$message = @(
        "[TAS] Update available: `$LatestTag (installed v`$InstalledVersion)",
        "[TAS] Download: `$ReleaseUrl",
        ""
    ) -join [Environment]::NewLine

    Set-Content -Path `$messagePath -Value `$message -Encoding Ascii
}

if (Test-Path `$lockPath) {
    `$lockAge = (Get-Date).ToUniversalTime() - (Get-Item `$lockPath).LastWriteTimeUtc.ToUniversalTime()
    if (`$lockAge -lt [TimeSpan]::FromMinutes(5)) {
        exit 0
    }

    Remove-Item -Path `$lockPath -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType File -Path `$lockPath -Force | Out-Null

try {
    `$state = `$null
    if (Test-Path `$statePath) {
        try {
            `$state = Get-Content -Path `$statePath -Raw | ConvertFrom-Json
        }
        catch {
            `$state = `$null
        }
    }

    if (`$state -and `$state.LastCheckedUtc) {
        try {
            `$lastChecked = [DateTime]::Parse(`$state.LastCheckedUtc, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::RoundtripKind)
            if (((Get-Date).ToUniversalTime() - `$lastChecked.ToUniversalTime()) -lt `$cooldown) {
                exit 0
            }
        }
        catch {
        }
    }

    `$installedVersion = Get-InstalledVersion
    if (-not `$installedVersion) {
        exit 0
    }

    `$headers = @{
        "Accept"     = "application/vnd.github+json"
        "User-Agent" = "TheAnimeScripter-Version-Check"
    }

    `$release = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/`$Repository/releases/latest" -Headers `$headers
    `$latestTag = if (`$release.tag_name) { `$release.tag_name } else { "" }
    `$latestVersion = `$latestTag.TrimStart('v', 'V')
    `$updateAvailable = `$false

    if (-not [string]::IsNullOrWhiteSpace(`$latestVersion)) {
        try {
            `$updateAvailable = ([version]`$latestVersion -gt [version]`$installedVersion)
        }
        catch {
            `$updateAvailable = (`$latestVersion -ne `$installedVersion)
        }
    }

    if (`$updateAvailable) {
        Set-UpdateMessage -InstalledVersion `$installedVersion -LatestTag `$latestTag -ReleaseUrl `$release.html_url
    }
    else {
        Remove-UpdateMessage
    }

    [pscustomobject]@{
        LastCheckedUtc   = (Get-Date).ToUniversalTime().ToString("o")
        InstalledVersion = `$installedVersion
        LatestTag        = `$latestTag
        LatestVersion    = `$latestVersion
        UpdateAvailable  = `$updateAvailable
        ReleaseUrl       = `$release.html_url
    } | ConvertTo-Json | Set-Content -Path `$statePath -Encoding Ascii
}
catch {
}
finally {
    Remove-Item -Path `$lockPath -Force -ErrorAction SilentlyContinue
}
"@

        $launcherContent = @"
@echo off
setlocal
set "TAS_ROOT=%~dp0"
set "TAS_NOTICE_FILE=%TAS_ROOT%tas-update-message.txt"
set "TAS_CHECKER=%TAS_ROOT%tas-version-check.ps1"
set "TAS_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if exist "%TAS_NOTICE_FILE%" type "%TAS_NOTICE_FILE%"
if exist "%TAS_CHECKER%" if exist "%TAS_POWERSHELL%" start "" /B "%TAS_POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%TAS_CHECKER%"

"%TAS_ROOT%python.exe" "%TAS_ROOT%main.py" %*
"@

        Set-Content -Path (Join-Path $InstallPath "tas-version-check.ps1") -Value $versionCheckerContent -Encoding Ascii
        Set-Content -Path (Join-Path $InstallPath "TheAnimeScripter.cmd") -Value $launcherContent -Encoding Ascii
        Set-Content -Path (Join-Path $InstallPath "tas.cmd") -Value $launcherContent -Encoding Ascii

        $requiredFiles = @(
            (Join-Path $InstallPath "python.exe"),
            (Join-Path $InstallPath "main.py"),
            (Join-Path $InstallPath "tas-version-check.ps1"),
            (Join-Path $InstallPath "TheAnimeScripter.cmd"),
            (Join-Path $InstallPath "tas.cmd")
        )

        $missingFiles = @($requiredFiles | Where-Object { -not (Test-Path $_) })
        if ($missingFiles.Count -gt 0) {
            throw "Installation completed with missing files: $($missingFiles -join ', ')"
        }

        if ($shouldAddToPath) {
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            $userEntries = @()
            if (-not [string]::IsNullOrWhiteSpace($userPath)) {
                $userEntries = @($userPath -split ";")
            }

            if (-not (Test-PathEntryPresent -Entries $userEntries -Candidate $InstallPath)) {
                $newUserPath = if ($userEntries.Count -gt 0) {
                    ($userEntries + $InstallPath) -join ";"
                }
                else {
                    $InstallPath
                }

                [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")

                if (-not (Test-PathEntryPresent -Entries ($env:Path -split ";") -Candidate $InstallPath)) {
                    $env:Path = "$InstallPath;$env:Path"
                }

                Write-Success "Added '$InstallPath' to the user PATH. Open a new shell if the commands are not visible yet."
            }
            else {
                Write-Info "The install path is already present in the user PATH."
            }
        }

        $tagName = if ($release.tag_name) { $release.tag_name } else { "latest" }
        Write-Success "Installed TheAnimeScripter $tagName to $InstallPath"
        Write-Host ""
        Write-Host "From this directory:" -ForegroundColor Yellow
        Write-Host "  .\TheAnimeScripter.cmd -h"
        Write-Host "  .\tas.cmd -h"

        if ($shouldAddToPath) {
            Write-Host ""
            Write-Host "Global commands in this shell:" -ForegroundColor Yellow
            Write-Host "  TheAnimeScripter -h"
            Write-Host "  tas -h"
        }
    }
    finally {
        if ($temporaryRoot -and (Test-Path $temporaryRoot)) {
            Remove-Item -Path $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
} @args