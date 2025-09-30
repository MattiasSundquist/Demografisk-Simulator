# --- Configuration ---
$ProjectRoot = "." # Current directory. This is less critical with the hardcoded trim.
$OutputFile = "project_snapshot.txt"
$IgnoreDirs = @(
    "venv", "__pycache__"
)
$IgnoreFiles = @(
    $OutputFile
)
$IncludeExtensions = @(
    ".py", ".md"
)
# The specific string to find and trim before (your project's root folder name)
$PathAnchor = "Statistik" # Ensure this is the exact name of the folder
# --- End of Configuration ---

$AbsoluteProjectRoot = Resolve-Path -LiteralPath $ProjectRoot # Still useful for Get-ChildItem
$AbsoluteOutputFile = Join-Path -Path $AbsoluteProjectRoot -ChildPath $OutputFile

New-Item -ItemType File -Path $AbsoluteOutputFile -Force | Out-Null
Write-Host "Starting code snapshot generation..."
Write-Host "Output will be saved to: $AbsoluteOutputFile"

Get-ChildItem -Path $AbsoluteProjectRoot -Recurse -File | ForEach-Object {
    $file = $_
    $fullPathNormalized = $file.FullName -replace "\\", "/" # Normalize to forward slashes first

    # Find the anchor string (case-insensitive) and get the path part after it
    $anchorWithSlash = "$PathAnchor/"
    $indexOfAnchor = $fullPathNormalized.IndexOf($anchorWithSlash, [System.StringComparison]::OrdinalIgnoreCase)

    $displayPath = $fullPathNormalized # Default to full path if anchor not found

    if ($indexOfAnchor -ge 0) {
        # Get the substring starting AFTER the anchor and its trailing slash
        $displayPath = $fullPathNormalized.Substring($indexOfAnchor + $anchorWithSlash.Length)
    } else {
        # If "AISiteForge/" is not found, it's unexpected.
        # For safety, we'll try to make it relative to the script's $ProjectRoot as a fallback,
        # or just use the filename if all else fails.
        Write-Warning "Warning: Anchor '$anchorWithSlash' not found in path '$fullPathNormalized'. Path may not be trimmed as expected."
        # Fallback: try to make it relative to the script-defined root
        if ($fullPathNormalized.StartsWith(($AbsoluteProjectRoot.FullName -replace "\\", "/") + "/", [System.StringComparison]::OrdinalIgnoreCase)) {
            $tempRelative = $fullPathNormalized.Substring(($AbsoluteProjectRoot.FullName -replace "\\", "/").Length).TrimStart('/')
            $displayPath = $tempRelative
        } else {
            $displayPath = $file.Name # Last resort, just the file name
        }
    }

    if ($file.FullName -eq $AbsoluteOutputFile) { return }

    $isInIgnoredDir = $false
    foreach ($ignoreDir in $IgnoreDirs) {
        # This is your original ignore logic. It might need adjustment
        # if $PathAnchor changes how paths are perceived by this logic.
        # For now, keeping it as you provided.
        if (($file.DirectoryName -eq $AbsoluteProjectRoot.FullName -and $file.Directory.Name -eq $ignoreDir) `
            -or ($file.DirectoryName + "\" -like "*\$ignoreDir\*") `
            -or ($file.DirectoryName -like "*\$ignoreDir")) {
            $isInIgnoredDir = $true; break
        }
    }
    if ($isInIgnoredDir) { return }
    if ($IgnoreFiles -contains $file.Name) { return }

    if ($IncludeExtensions -contains $file.Extension.ToLowerInvariant()) {
        Write-Host "Processing: $displayPath"
        Add-Content -Path $AbsoluteOutputFile -Value "--- START OF FILE: $displayPath ---"
        try {
            $content = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
            Add-Content -Path $AbsoluteOutputFile -Value $content
        }
        catch {
            Add-Content -Path $AbsoluteOutputFile -Value "Error reading file $($file.FullName): $($_.Exception.Message)"
            Write-Warning "Error reading file '$($file.FullName)': $($_.Exception.Message)"
        }
        Add-Content -Path $AbsoluteOutputFile -Value "`n--- END OF FILE: $displayPath ---`n"
    }
}

Write-Host "Code snapshot generation complete: $AbsoluteOutputFile"
Write-Host "Press any key to continue..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null