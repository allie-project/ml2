# Dumb, stupid, dumb hack for Ninja to find Windows SDK include files using vcvars.

$VSSearchPaths = @(
	'E:\Programs\Visual Studio\2022\Community\Common7\Tools',
	'E:\Programs\Visual Studio\2019\Community\Common7\Tools',
	'E:\Programs\Visual Studio\2017\Community\Common7\Tools',
	'C:\Program Files (x86)\Visual Studio\2022\Community\Common7\Tools',
	'C:\Program Files (x86)\Visual Studio\2019\Community\Common7\Tools',
	'C:\Program Files (x86)\Visual Studio\2017\Community\Common7\Tools',
	'C:\Program Files\Visual Studio\2022\Community\Common7\Tools',
	'C:\Program Files\Visual Studio\2019\Community\Common7\Tools',
	'C:\Program Files\Visual Studio\2017\Community\Common7\Tools',
	'E:\Program Files (x86)\Visual Studio\2022\Community\Common7\Tools',
	'E:\Program Files (x86)\Visual Studio\2019\Community\Common7\Tools',
	'E:\Program Files (x86)\Visual Studio\2017\Community\Common7\Tools',
	'E:\Program Files\Visual Studio\2022\Community\Common7\Tools',
	'E:\Program Files\Visual Studio\2019\Community\Common7\Tools',
	'E:\Program Files\Visual Studio\2017\Community\Common7\Tools'
);

foreach ($VSSearchPath in $VSSearchPaths) {
	if (Test-Path $VSSearchPath) {
		$IncludePath = $VSSearchPath;
		break;
	}
}

if (!$IncludePath -or -not (Test-Path "$IncludePath\VsDevCmd.bat")) {
	throw "Could not find Visual Studio path!";
}

$Architecture = $args[0];
if (!$args[0]) {
	if (-not $env:_LIBGLIDE_VCVARS_COMPLETE) {
		Write-Host "Assuming 'amd64' architecture. Pass 'x86', 'arm64', or 'arm' as the first argument to override." -ForegroundColor Yellow;
	}

	$Architecture = 'amd64';
}

if ($env:_LIBGLIDE_VCVARS_COMPLETE -and ($env:_LIBGLIDE_VCVARS_COMPLETE -ne $Architecture)) {
	throw "Please restart your PowerShell instance to change architectures.";
} elseif ($env:_LIBGLIDE_VCVARS_COMPLETE -eq $Architecture) {
	throw "Visual Studio already set up for $Architecture.";
}

Push-Location $IncludePath;

cmd.exe /c "VsDevCmd.bat -arch=$Architecture&set" |
ForEach-Object {
	if ($_ -match "=") {
		$v = $_.split("=", 2);
		Set-Item -Force -Path "ENV:\$($v[0])" -Value "$($v[1])";
	}
}

Pop-Location;

$env:_LIBGLIDE_VCVARS_COMPLETE = $Architecture;