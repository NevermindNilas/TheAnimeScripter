var mainWindow = new Window("palette", "AnimeScripter", undefined);
    mainWindow.orientation = "column";
    mainWindow.alignChildren = ["left", "top"];
    mainWindow.spacing = 20;
    if (mainWindow !== null) {
        var interpolationGroup = mainWindow.add("group");
        var interpolationButton = interpolationGroup.add("button", undefined, "Interpolation");
        var interpolationDropdown = interpolationGroup.add("dropdownlist", undefined, ["2x", "4x"]);
        interpolationDropdown.selection = 0;
 
        var upscaleGroup = mainWindow.add("group");
        var upscaleButton = upscaleGroup.add("button", undefined, "  Upscale  ");
        var upscaleDropdown = upscaleGroup.add("dropdownlist", undefined, ["2x", "3x", "4x"]);
        upscaleDropdown.selection = 0;
 
        var dedupGroup = mainWindow.add("group");
        var dedupButton = dedupGroup.add("button", undefined, "Dedup");
        var dedupDropdown = dedupGroup.add("dropdownlist", undefined, ["FFMPEG", "SSIM ( N / A)", "VMAF ( N / A )", "Hash ( N / A )"]);
        dedupDropdown.selection = 0;
 
        var settingsButtonGroup = mainWindow.add("group");
        var settingsButton = settingsButtonGroup.add("button", undefined, "Settings");
        settingsButton.onClick = function () {
            var newWindow = new Window("dialog", "Settings", undefined);
            newWindow.orientation = "column";
            newWindow.alignChildren = ["left", "top"];
            newWindow.spacing = 10;
            if ( newWindow !== null ) {
                var InterpolationGroup = newWindow.add("group");
                var InterpolationButton = InterpolationGroup.add("statictext", undefined, "Choose Interpolation");
                var InterpolationDropdown = InterpolationGroup.add("dropdownlist", undefined, ["Rife Cuda (fast)", "Rife AMD ( N / A) (slower)", "Rife CPU (slowest)"]);
                InterpolationDropdown.selection = 0;

                var UpscaleGroup = newWindow.add("group");
                var UpscaleButton = UpscaleGroup.add("statictext", undefined, "Choose Upscale");
                var UpscaleDropdown = UpscaleGroup.add("dropdownlist", undefined, ["ShuffleCugan ( fastest )", "Cugan ( fast )", "SwinIR ( slow )", "ESRGAN ( N / A)", "RealSR ( N / A )"]);
                UpscaleDropdown.selection = 0;

                var CuganOptions = newWindow.add("group");
                var CuganOptionsButton = CuganOptions.add("statictext", undefined, "Cugan Threads");
                var CuganOptionsDropdown = CuganOptions.add("dropdownlist", undefined, ["1    ", "2", "3", "4"]);
                CuganOptionsDropdown.selection = 1;
            }
            newWindow.show();
        }
} 
if ( mainWindow instanceof Window) {
    mainWindow.center();
    mainWindow.show();
} 
else  {
    mainWindow.layout.layout(true);
}