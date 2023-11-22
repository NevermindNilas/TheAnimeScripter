var mainWindow = new Window("palette", "AnimeScripter", undefined);
if (mainWindow !== null) { 
    var interpolationGroup = mainWindow.add("group");
    interpolationGroup.add("statictext", undefined, "Interpolation:");
    var interpolationDropdown = interpolationGroup.add("dropdownlist", undefined, ["2x", "4x"]);
    interpolationDropdown.selection = 0;

    var upscaleGroup = mainWindow.add("group");
    upscaleGroup.add("statictext", undefined, "Upscale:");
    var upscaleDropdown = upscaleGroup.add("dropdownlist", undefined, ["2x", "3x", "4x"]);
    upscaleDropdown.selection = 0;

    var dedupGroup = mainWindow.add("group");
    dedupGroup.add("statictext", undefined, "Dedup:");
    var dedupDropdown = dedupGroup.add("dropdownlist", undefined, ["FFMPEG", "SSIM ( N / A)", "VMAF ( N / A )"]);
    dedupDropdown.selection = 0;

    var settingsButtonGroup = mainWindow.add("group");
    var settingsButton = settingsButtonGroup.add("button", undefined, "Settings");

    var startButtonGroup = mainWindow.add("group");
    var startButton = startButtonGroup.add("button", undefined, "Start");
}

if (mainWindow instanceof Window) { 
    mainWindow.center();
    mainWindow.show();
}
else {
    mainWindow.layout.layout(true);
}