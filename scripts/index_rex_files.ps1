param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [string]$PythonPath = 'python',
    [ValidateSet('rex','ct')][string[]]$Sources = @('rex'),
    [ValidateRange(1,5)][int]$MaxAttempts = 3,
    [ValidateRange(5,120)][int]$TimeoutSec = 30
)
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Windows PowerShell 5.1 may otherwise negotiate an obsolete TLS version.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Get-SafeFailure($Record) {
    $failure = $Record.Exception
    $parts = @()
    while ($null -ne $failure) {
        $message = $failure.Message -replace 'hf_[A-Za-z0-9]+','[REDACTED]' -replace '(?i)Bearer\s+\S+','Bearer [REDACTED]'
        $parts += ($failure.GetType().Name + ': ' + $message)
        $failure = $failure.InnerException
    }
    return ($parts -join ' -> ')
}

function Get-RepositoryIndex([string]$Repo, [string]$Revision, [hashtable]$Headers) {
    # Sizes are useful but optional. Fall back to filenames if blobs expansion fails.
    $baseUri = 'https://huggingface.co/api/datasets/' + $Repo
    $endpoints = @($baseUri + '/revision/' + $Revision + '?blobs=true';
                   $baseUri + '/revision/' + $Revision;
                   $baseUri)
    $lastFailure = ''
    foreach ($endpoint in $endpoints) {
        for ($attempt=1; $attempt -le $MaxAttempts; $attempt++) {
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri $endpoint -Headers $Headers -TimeoutSec $TimeoutSec
                $info = $response.Content | ConvertFrom-Json
                if ($info.sha -ne $Revision) {
                    throw 'Repository revision differs from the saved metadata; refusing to mix snapshots.'
                }
                if ($null -eq $info.siblings -or @($info.siblings).Count -eq 0) {
                    throw 'Repository response does not contain a file list.'
                }
                return $info
            } catch {
                $lastFailure = Get-SafeFailure $_
                $status = if ($null -ne $_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { $null }
                # Authentication/access errors will not improve by retrying.
                if ($status -eq 401 -or $status -eq 403) {
                    throw ('HTTP ' + $status + ': check token permissions and dataset access for ' + $Repo)
                }
                Write-Warning ('Index request ' + $attempt + '/' + $MaxAttempts + ' failed: ' + $lastFailure)
                if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds ([Math]::Min(2 * $attempt, 6)) }
            }
        }
        Write-Warning 'Trying the next repository endpoint.'
    }
    throw ('All repository endpoints failed. ' + $lastFailure)
}
$hfAccessToken = & $PythonPath -c "from huggingface_hub import get_token; print(get_token() or '')"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($hfAccessToken)) { throw 'Run hf auth login first.' }
$hfHeaders = @{ Authorization = 'Bearer ' + $hfAccessToken.Trim() }
$revisions = Get-Content -LiteralPath (Join-Path $DataRoot 'manifests/revisions.json') -Raw -Encoding utf8 | ConvertFrom-Json
try {
    foreach ($source in $Sources) {
        $repo = $revisions.($source + '_repo')
        $revision = $revisions.($source + '_revision')
        Write-Output ('Indexing ' + $source + ' at revision ' + $revision)
        $info = Get-RepositoryIndex -Repo $repo -Revision $revision -Headers $hfHeaders
        $target = Join-Path $DataRoot ('manifests/' + $source + '_files.json')
        $partial = $target + '.partial'
        @($info.siblings) | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $partial -Encoding utf8
        Move-Item -LiteralPath $partial -Destination $target -Force
        Write-Output ($source + ': indexed ' + @($info.siblings).Count + ' files')
    }
} catch {
    [Console]::Error.WriteLine('Repository indexing failed for ' + $source + ': ' + (Get-SafeFailure $_))
    exit 1
} finally { $hfAccessToken=$null; $hfHeaders=$null }
