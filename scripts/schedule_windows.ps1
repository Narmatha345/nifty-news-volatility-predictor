# Registers two Windows scheduled tasks for the CURRENT user (runs only while you are logged on,
# no password or credential is stored):
#
#   NiftyNews-PreMarket   weekdays 08:30 IST  python -m backend.cli daily --validate  (collect -> predict -> LLM review)
#   NiftyNews-AfterClose  weekdays 16:30 IST  python -m backend.cli outcomes   (prices -> actual outcomes)
#
# Times are given in IST and converted to this machine's time zone. NSE holidays need no special
# handling: `daily` always targets the next session whose 09:15 IST open is still ahead, and
# `outcomes` only scores sessions that have closed.
#
# Usage (from the project root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_windows.ps1            # create / update
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_windows.ps1 -Remove    # delete both tasks
param([switch]$Remove, [string]$PreMarketIST = "08:30", [string]$AfterCloseIST = "16:30")

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data"
$names = @("NiftyNews-PreMarket", "NiftyNews-AfterClose")

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Output "Removed: $($names -join ', ')"
    exit 0
}
if (-not (Test-Path $python)) { throw "Python venv not found at $python" }

$ist = [System.TimeZoneInfo]::FindSystemTimeZoneById("India Standard Time")
function Local-From-IST([string]$hhmm) {
    $t = [datetime]::ParseExact($hhmm, "HH:mm", $null)
    $istToday = [datetime]::SpecifyKind([datetime]::Today.Add($t.TimeOfDay), [System.DateTimeKind]::Unspecified)
    return [System.TimeZoneInfo]::ConvertTime($istToday, $ist, [System.TimeZoneInfo]::Local)
}

$jobs = @(
    @{ Name = $names[0]; At = (Local-From-IST $PreMarketIST); Cmd = "daily --validate"; Log = "daily.log" },
    @{ Name = $names[1]; At = (Local-From-IST $AfterCloseIST); Cmd = "outcomes"; Log = "outcomes.log" }
)
foreach ($j in $jobs) {
    $arg = "/c cd /d `"$root`" && `"$python`" -m backend.cli $($j.Cmd) >> `"$logDir\$($j.Log)`" 2>&1"
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $arg
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $j.At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Force | Out-Null
    Write-Output ("{0}: weekdays at {1:HH:mm} local ({2} IST) -> backend.cli {3}" -f $j.Name, $j.At,
        $(if ($j.Cmd -like "daily*") { $PreMarketIST } else { $AfterCloseIST }), $j.Cmd)
}
Write-Output "Note: StartWhenAvailable runs a missed pre-market job late; a late 'daily' targets the NEXT session."
