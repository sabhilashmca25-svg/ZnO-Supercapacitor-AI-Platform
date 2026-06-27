' ZnO Supercapacitor AI Platform -- Desktop launcher
' Calls run_launcher.bat via cmd /c.
' Using a batch intermediary avoids VBScript quoting issues with
' paths that contain spaces (e.g. "2nd semester").
' Window style 7 = minimised, no focus steal.

Dim fso, root, bat
Set fso = CreateObject("Scripting.FileSystemObject")
root    = fso.GetParentFolderName(WScript.ScriptFullName)
bat     = root & "\run_launcher.bat"

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c """ & bat & """", 1, False
