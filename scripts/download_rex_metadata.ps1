param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [string]$PythonPath = 'python'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Capture credentials in memory; never echo them or include them in command arguments.
$hfAccessToken = & $PythonPath -c "from huggingface_hub import get_token; print(get_token() or '')"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($hfAccessToken)) {
    throw 'No cached Hugging Face token. Run hf auth login first.'
}
$hfHeaders = @{ Authorization = 'Bearer ' + $hfAccessToken.Trim() }
$sources = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../configs/rex_sources.json') -Raw -Encoding utf8 | ConvertFrom-Json
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
try {
    $stage = 'local metadata directory'
    $rexFolder = Join-Path $DataRoot 'raw/rexgroundingct'
    $manifestFolder = Join-Path $DataRoot 'manifests'
    New-Item -ItemType Directory -Path $rexFolder,$manifestFolder -Force | Out-Null
    foreach ($file in @('dataset.json','reports_dataset.json')) {
        $stage = $file
        $uri = 'https://huggingface.co/datasets/' + $sources.rex_repo + '/resolve/' + $sources.rex_revision + '/' + $file
        $destination = Join-Path $rexFolder $file
        Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers $hfHeaders -OutFile ($destination + '.partial') -TimeoutSec 120
        # Reject incomplete or non-JSON responses before replacing a previous download.
        Get-Content -LiteralPath ($destination + '.partial') -Raw -Encoding utf8 | ConvertFrom-Json | Out-Null
        Move-Item -LiteralPath ($destination + '.partial') -Destination $destination -Force
        Write-Output ('Metadata ready: ' + $file)
    }
    $sources | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $manifestFolder 'revisions.json') -Encoding utf8
    Write-Output 'Metadata download complete.'
} catch {
    # Avoid echoing signed URLs, request headers or authentication details.
    $status = if ($null -ne $_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 'no HTTP response' }
    Write-Error ('Hugging Face metadata download failed at ' + $stage + ': ' + $_.Exception.GetType().Name + ' (' + $status + ')')
    exit 1
} finally {
    $hfAccessToken = $null
    $hfHeaders = $null
}
