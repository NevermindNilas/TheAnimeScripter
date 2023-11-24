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
        var upscaleButton = upscaleGroup.add("button", undefined, "Upscale");
        var upscaleDropdown = upscaleGroup.add("dropdownlist", undefined, ["2x", "3x", "4x"]);
        upscaleDropdown.selection = 0;
 
        var dedupGroup = mainWindow.add("group");
        var dedupButton = dedupGroup.add("button", undefined, "Dedup");
        var dedupDropdown = dedupGroup.add("dropdownlist", undefined, ["FFMPEG", "SSIM ( N / A)", "VMAF ( N / A )", "Hash ( N / A )"]);
        dedupDropdown.selection = 0;

        var segmentGroup = mainWindow.add("group");
        var segmentButton = segmentGroup.add("button", undefined, "Segment");
        var segmentDropdown = segmentGroup.add("dropdownlist", undefined, ["ISRNET", "Something", "Something", "Something"]);
        segmentDropdown.selection = 0;

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
                var InterpolationDropdown = InterpolationGroup.add("dropdownlist", undefined, ["Rife NVIDIA (fastest)", "Rife AMD ( N / A) (slow)", "Rife CPU (slowest)"]);
                InterpolationDropdown.selection = 0;

                var UpscaleGroup = newWindow.add("group");
                var UpscaleButton = UpscaleGroup.add("statictext", undefined, "Choose Upscale");
                var UpscaleDropdown = UpscaleGroup.add("dropdownlist", undefined, ["ShuffleCugan (fastest)", "Ultracompact (faster)", "Compact (fast)", "Cugan (fast)", "SwinIR (slow)"]);
                UpscaleDropdown.selection = 0;

                var CuganOptions = newWindow.add("group");
                var CuganOptionsButton = CuganOptions.add("statictext", undefined, "Upscale Threads");
                var CuganOptionsDropdown = CuganOptions.add("dropdownlist", undefined, ["1", "2", "3", "4"]);
                CuganOptionsDropdown.selection = 1;
            }
            newWindow.show();
        }

        var infoGroup = mainWindow.add("group");
        var infoButton = infoGroup.add("button", undefined, "Info");
        infoButton.onClick = function () {
            var newWindow = new Window("dialog", "Info", undefined);
            newWindow.orientation = "column";
            newWindow.alignChildren = ["left", "top"];
            newWindow.spacing = 10;

            if newWindow !== null {
                var InfoText = newWindow.add("statictext", undefined,"
                -video :str      - Takes full path of input file.

                -model_type :str - Can be Rife, Cugan, ShuffleCugan, Compact(N/A), SwinIR, Dedup, Segment (N/A), UltraCompact (N/A).

                -half :bool      - Set to True by default, utilizes FP16, more performance for free generally.

                -multi :int      - Used by both Upscaling and Interpolation, 
                                Cugan can utilize scale from 2-4,
                                Shufflecugan only 2, 
                                Compact 2,
                                SwinIR 2 or 4, 
                                Rife.. virtually anything.

                -kind_model :str - Cugan: no-denoise, conservative, denoise1x, denoise2x, denoise3x
                                SwinIR: small, medium, large.
                                Dedup: ffmpeg, Hash(N/A), VMAF(N/A), SSIM(N/A)

                -pro :bool       - Set to False by default, Only for CUGAN, utilize pro models.

                -nt :int         - Number of threads to utilize for Upscaling and Segmentation,
                                Really CPU/GPU dependent, with my 3090 I max out at 4 for Cugan / Shufflecugan.
                                As for SwinIR I max out at 2"
            }
        }
} 
if ( mainWindow instanceof Window) {
    mainWindow.center();
    mainWindow.show();
} 
else  {
    mainWindow.layout.layout(true);
}