param(
    [string]$PythonExe = "C:\qlik\tools\apache-tomcat-10.1.50-instance1\temp\Kronos\.venv\Scripts\python.exe",
    [string]$TargetDir = "C:\qlik\tools\kronos-qlib-toolkit\data\qlib_cn"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -m pip install pyqlib
& $PythonExe -m qlib.run.get_data qlib_data --target_dir $TargetDir --region cn
