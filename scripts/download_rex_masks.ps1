param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [ValidateRange(0,5000)][int]$Limit = 0,
    [ValidateRange(1,8)][int]$Workers = 1,
    [int]$WorkerIndex = -1,
    [string]$PythonPath = 'python'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($Workers -gt 1 -and $WorkerIndex -eq -1) {
    $active = @(for ($i=0; $i -lt $Workers; $i++) {
        Start-Job -ArgumentList $PSCommandPath,$DataRoot,$Limit,$Workers,$i,$PythonPath -ScriptBlock {
            param($ScriptFile,$Root,$Top,$Count,$Index,$Python)
            & $ScriptFile -DataRoot $Root -Limit $Top -Workers $Count -WorkerIndex $Index -PythonPath $Python
            if ($LASTEXITCODE -ne 0) { throw "Worker $Index exited with code $LASTEXITCODE" }
        }
    })
    try {
        while (@($active | Where-Object State -in 'Running','NotStarted').Count) {
            Receive-Job -Job $active -ErrorAction Continue
            Start-Sleep -Seconds 2
        }
        Receive-Job -Job $active -ErrorAction Continue
        if (@($active | Where-Object State -ne 'Completed').Count) { exit 1 }
        exit 0
    } finally { Remove-Job -Job $active -Force }
}
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$plan = Get-Content -LiteralPath (Join-Path $DataRoot 'manifests/rex_mask_plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($plan.repo -ne 'rajpurkarlab/ReXGroundingCT' -or $plan.revision -cnotmatch '^[0-9a-f]{40}$') {
    throw 'Invalid pinned ReX repository or revision.'
}
$token = & $PythonPath -c "from huggingface_hub import get_token; print(get_token() or '')"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($token)) { throw 'Run hf auth login first.' }
$headers = @{ Authorization = 'Bearer ' + $token.Trim() }
$root = Join-Path $DataRoot 'raw/rexgroundingct/segmentations'
New-Item -ItemType Directory -Path $root -Force | Out-Null

function Test-MaskFile($Path, $Job) {
    return ((Test-Path -LiteralPath $Path -PathType Leaf) -and
        (Get-Item -LiteralPath $Path).Length -eq $Job.size -and
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Job.sha256)
}

$verified = 0
$failed = 0
$jobs = @($plan.jobs)
if ($Limit -gt 0) { $jobs = @($jobs | Select-Object -First $Limit) }
if ($WorkerIndex -ge 0) {
    if ($WorkerIndex -ge $Workers) { throw 'Worker index is outside the worker count.' }
    $jobs = @(for ($i=$WorkerIndex; $i -lt $jobs.Count; $i+=$Workers) { $jobs[$i] })
}
foreach ($job in $jobs) {
    if ($job.path -cnotmatch '^segmentations/([a-z]+_\d+_[a-z]_\d+\.nii\.gz)$') { throw 'Unexpected ReX mask path.' }
    $target = Join-Path $root $Matches[1]
    if (!(Test-MaskFile $target $job)) {
        $success = $false
        $uri = 'https://huggingface.co/datasets/' + $plan.repo + '/resolve/' + $plan.revision + '/' + $job.path
        for ($attempt=1; $attempt -le 3; $attempt++) {
            try {
                $partial = $target + '.partial'
                Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers $headers -OutFile $partial -TimeoutSec 35
                if (!(Test-MaskFile $partial $job)) { throw 'File size or SHA-256 did not match.' }
                Move-Item -LiteralPath $partial -Destination $target -Force
                $success = $true
                break
            } catch {
                $status = if ($null -ne $_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { $null }
                if ($status -in 401,403) { throw ('ReX access rejected (HTTP ' + $status + ').') }
                Write-Warning ('Attempt ' + $attempt + ' failed: ' + $_.Exception.GetType().Name + ' HTTP ' + $status)
                if ($attempt -lt 3) { Start-Sleep -Seconds $attempt }
            }
        }
        if (!$success) {
            $failed++
            Write-Warning ('Mask unavailable in this run: ' + $job.volume_name)
            if ($failed -ge 3 -and $verified -eq 0) { break }
            continue
        }
    }
    $verified++
    if ($verified -eq 1 -or $verified % 50 -eq 0) { Write-Output ('Verified ' + $verified + '/' + $jobs.Count + ' ReX masks') }
}
Write-Output ('Verified ' + $verified + '/' + $jobs.Count + ' ReX masks. Failed ' + $failed)
if ($verified -ne $jobs.Count) { exit 1 }
exit 0
