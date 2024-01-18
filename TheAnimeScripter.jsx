var panelGlobal = this;
var TheAnimeScripter = (function() {

    var scriptName = "TheAnimeScripter";

    /*
    scriptVersion = "0.1.7";
    scriptAuthor = "Nilas";
    scriptURL = "https://github.com/NevermindNilas/TheAnimeScripter"
    discordServer = "https://discord.gg/CdRD9GwS8J"
    */

    // Default Values for the settings
    var outputFolder = app.settings.haveSetting(scriptName, "outputFolder") ? app.settings.getSetting(scriptName, "outputFolder") : "undefined";
    var TheAnimeScripterPath = app.settings.haveSetting(scriptName, "TheAnimeScripterPath") ? app.settings.getSetting(scriptName, "TheAnimeScripterPath") : "undefined";
    var dropdownModel = app.settings.haveSetting(scriptName, "dropdownModel") ? app.settings.getSetting(scriptName, "dropdownModel") : 0;
    var dropdownCugan = app.settings.haveSetting(scriptName, "dropdownCugan") ? app.settings.getSetting(scriptName, "dropdownCugan") : 0;
    var dropdownSwinIr = app.settings.haveSetting(scriptName, "dropdownSwinIr") ? app.settings.getSetting(scriptName, "dropdownSwinIr") : 0;
    var dropdwonSegment = app.settings.haveSetting(scriptName, "dropdwonSegment") ? app.settings.getSetting(scriptName, "dropdwonSegment") : 0;
    var intInterpolate = app.settings.haveSetting(scriptName, "intInterpolate") ? app.settings.getSetting(scriptName, "intInterpolate") : 2;
    var intUpscale = app.settings.haveSetting(scriptName, "intUpscale") ? app.settings.getSetting(scriptName, "intUpscale") : 2;
    var sliderSharpen = app.settings.haveSetting(scriptName, "sliderSharpen") ? app.settings.getSetting(scriptName, "sliderSharpen") : 50;
    var dropdownDedupStrenght = app.settings.haveSetting(scriptName, "dropdownDedupStrenght") ? app.settings.getSetting(scriptName, "dropdownDedupStrenght") : 0
    var sliderSceneChange = app.settings.haveSetting(scriptName, "sliderSceneChange") ? app.settings.getSetting(scriptName, "sliderSceneChange") : 70;
    var dropdownEncoder = app.settings.haveSetting(scriptName, "dropdownEncoder") ? app.settings.getSetting(scriptName, "dropdownEncoder") : 0;
    var dropdownInterpolate = app.settings.haveSetting(scriptName, "dropdownInterpolate") ? app.settings.getSetting(scriptName, "dropdownInterpolate") : 0;
    var sliderDedupSenstivity = app.settings.haveSetting(scriptName, "sliderDedupSenstivity") ? app.settings.getSetting(scriptName, "sliderDedupSenstivity") : 50;

    var segmentValue = 0;
    var sceneChangeValue = 0;
    var depthValue = 0;
    var motionBlurValue = 0;

    // THEANIMESCRIPTER
    // ================
    var TheAnimeScripter = (panelGlobal instanceof Panel) ? panelGlobal : new Window("palette");
    if (!(panelGlobal instanceof Panel)) TheAnimeScripter.text = "TheAnimeScripter";
    TheAnimeScripter.orientation = "column";
    TheAnimeScripter.alignChildren = ["center", "top"];
    TheAnimeScripter.spacing = 10;
    TheAnimeScripter.margins = 10;

    // PANELCHAIN
    // ==========
    var panelChain = TheAnimeScripter.add("panel", undefined, undefined, {
        name: "panelChain"
    });
    panelChain.text = "Chain";
    panelChain.orientation = "column";
    panelChain.alignChildren = ["left", "top"];
    panelChain.spacing = 10;
    panelChain.margins = 10;

    var buttonStartProcess = panelChain.add("button", undefined, undefined, {
        name: "buttonStartProcess"
    });
    buttonStartProcess.text = "Start  Process";
    buttonStartProcess.preferredSize.width = 104;
    buttonStartProcess.alignment = ["center", "top"];

    // GROUP1
    // ======
    var group1 = panelChain.add("group", undefined, {
        name: "group1"
    });
    group1.orientation = "row";
    group1.alignChildren = ["left", "center"];
    group1.spacing = 10;
    group1.margins = 0;

    var checkboxDeduplicate = group1.add("checkbox", undefined, "Deduplicate", {
        name: "checkboxDeduplicate"
    });
    checkboxDeduplicate.alignment = ["left", "center"];
    checkboxDeduplicate.helpTip = "Deduplicate using FFMPEG's mpdecimate filter";

    // GROUP2
    // ======
    var group2 = panelChain.add("group", undefined, {
        name: "group2"
    });
    group2.orientation = "row";
    group2.alignChildren = ["left", "center"];
    group2.spacing = 10;
    group2.margins = 0;

    var checkboxUpscale = group2.add("checkbox", undefined, "Upscale", {
        name: "checkboxUpscale"
    });
    checkboxUpscale.alignment = ["left", "center"];
    checkboxUpscale.helpTip = "Upscale using the model you choose";

    // GROUP3
    // ======
    var group3 = panelChain.add("group", undefined, {
        name: "group3"
    });
    group3.orientation = "row";
    group3.alignChildren = ["left", "center"];
    group3.spacing = 10;
    group3.margins = 0;

    var checkboxInterpolate = group3.add("checkbox", undefined, "Interpolate", {
        name: "checkboxInterpolate"
    });
    checkboxInterpolate.alignment = ["left", "center"];
    checkboxInterpolate.helpTip = "Interpolate using the selected model from the Dropdown";

    var group4 = panelChain.add("group", undefined, {
        name: "group4"
    });

    var checkboxSharpen = group4.add("checkbox", undefined, "Sharpen", {
        name: "checkboxSharpen"
    });
    checkboxSharpen.alignment = ["left", "center"];
    checkboxSharpen.helpTip = "Sharpen using Contrast Adaptive Sharpening";


    // panelPostProcess
    // ==========
    var panelPostProcess = TheAnimeScripter.add("panel", undefined, undefined, {
        name: "panelPostProcess"
    });
    panelPostProcess.text = "Post Process";
    panelPostProcess.orientation = "column";
    panelPostProcess.alignChildren = ["left", "top"];
    panelPostProcess.spacing = 10;
    panelPostProcess.margins = 10;

    var buttonDepthMap = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonDepthMap"
    });
    buttonDepthMap.text = "Depth Map";
    buttonDepthMap.preferredSize.width = 105;
    buttonDepthMap.alignment = ["center", "top"];

    var buttonSegment = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonSegment"
    });
    buttonSegment.text = "Rotobrush";
    buttonSegment.preferredSize.width = 105;
    buttonSegment.alignment = ["center", "top"];

    var buttonSceneChange = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonSceneChange"
    });

    buttonSceneChange.text = "Auto Cut";
    buttonSceneChange.preferredSize.width = 105;
    buttonSceneChange.alignment = ["center", "top"];

    var buttonMotionBlur = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonMotionBlur"
    });

    buttonMotionBlur.text = "Motion Blur";
    buttonMotionBlur.preferredSize.width = 105;
    buttonMotionBlur.alignment = ["center", "top"];
    buttonMotionBlur.helpTip = "Motion Blur using average weighted frame blending, use interpolation factor and model to determine how many frames to blend and the quality of the blending`";


    // PANELMORE
    // =========
    var panelMore = TheAnimeScripter.add("panel", undefined, undefined, {
        name: "panelMore"
    });
    panelMore.text = "More";
    panelMore.orientation = "column";
    panelMore.alignChildren = ["left", "top"];
    panelMore.spacing = 10;
    panelMore.margins = 10;

    var buttonInfo = panelMore.add("button", undefined, undefined, {
        name: "buttonInfo"
    });
    buttonInfo.text = "Info";
    buttonInfo.preferredSize.width = 105;
    buttonInfo.alignment = ["center", "top"];
    buttonInfo.enabled = false;

    var buttonSettings = panelMore.add("button", undefined, undefined, {
        name: "buttonSettings"
    });
    buttonSettings.text = "Settings";
    buttonSettings.preferredSize.width = 105;
    buttonSettings.alignment = ["center", "top"];

    TheAnimeScripter.layout.layout(true);
    TheAnimeScripter.layout.resize();
    TheAnimeScripter.onResizing = TheAnimeScripter.onResize = function() {
        this.layout.resize();
    }

    var settingsWindow = new Window("palette", undefined, undefined, {
        resizable: true,
        closeButton: false
    });

    settingsWindow.text = "Settings";
    settingsWindow.orientation = "column";
    settingsWindow.alignChildren = ["center", "top"];
    settingsWindow.spacing = 10;
    settingsWindow.margins = 10;

    // PANELONFIRSTRUN
    // ===============
    var panelOnFirstRun = settingsWindow.add("panel", undefined, undefined, {
        name: "panelOnFirstRun"
    });
    panelOnFirstRun.text = "On First Run";
    panelOnFirstRun.orientation = "column";
    panelOnFirstRun.alignChildren = ["left", "top"];
    panelOnFirstRun.spacing = 10;
    panelOnFirstRun.margins = 10;

    // GROUP1
    // ======
    var group1 = panelOnFirstRun.add("group", undefined, {
        name: "group1"
    });
    group1.orientation = "row";
    group1.alignChildren = ["left", "center"];
    group1.spacing = 10;
    group1.margins = 0;

    var buttonFolder = group1.add("button", undefined, undefined, {
        name: "buttonFolder"
    });
    buttonFolder.helpTip = "Set it to wherever The Anime Scripter folder is situated.";
    buttonFolder.text = "Set TAS Folder";
    buttonFolder.preferredSize.width = 100;

    var buttonOutput = group1.add("button", undefined, undefined, {
        name: "buttonOutput"
    });
    buttonOutput.text = "Set Output";
    buttonOutput.preferredSize.width = 101;
    buttonOutput.helpTip = "Set it to wherever you want the output to be saved.";

    // PANELDEFAULTPATHS
    // =================
    var panelDefaultPaths = settingsWindow.add("panel", undefined, undefined, {
        name: "panelDefaultPaths"
    });
    panelDefaultPaths.text = "Script Paths";
    panelDefaultPaths.orientation = "column";
    panelDefaultPaths.alignChildren = ["center", "top"];
    panelDefaultPaths.spacing = 10;
    panelDefaultPaths.margins = 10;
    panelDefaultPaths.preferredSize.width = 235;

    // Group 2

    var group2 = panelDefaultPaths.add("group", undefined, {
        name: "group2"
    });

    group2.orientation = "column";
    group2.alignChildren = ["center", "center"];
    group2.spacing = 10;
    group2.margins = 0;

    var labelTASPath = group2.add("statictext", undefined, "TAS Path:");
    var textTheAnimeScripterFolderValue = group2.add("statictext", undefined, undefined, {
        name: "textTheAnimeScripterFolderValue"
    });

    textTheAnimeScripterFolderValue.text = TheAnimeScripterPath;

    // Group 3

    var group3 = panelDefaultPaths.add("group", undefined, {
        name: "group3"
    });

    group3.orientation = "column";
    group3.alignChildren = ["center", "center"];
    group3.spacing = 10;
    group3.margins = 0;

    var labelOutputPath = group3.add("statictext", undefined, "Output Path:");
    var textOutputFolderValue = group3.add("statictext", undefined, undefined, {
        name: "textOutputFolderValue"
    });

    textOutputFolderValue.text = outputFolder;


    // GENERALPANEL
    // ============
    var generalPanel = settingsWindow.add("panel", undefined, undefined, {
        name: "generalPanel"
    });
    generalPanel.text = "General Settings";
    generalPanel.orientation = "column";
    generalPanel.alignChildren = ["left", "top"];
    generalPanel.spacing = 10;
    generalPanel.margins = 10;

    var textSharpen = generalPanel.add("statictext", undefined, undefined, {
        name: "textSharpen"
    });
    textSharpen.text = "Sharpenening Sensitivity";
    textSharpen.justify = "center";
    textSharpen.alignment = ["center", "top"];

    var sliderSharpen = generalPanel.add("slider", undefined, undefined, undefined, undefined, {
        name: "sliderSharpen"
    });
    sliderSharpen.minvalue = 0;
    sliderSharpen.maxvalue = 100;
    sliderSharpen.value = 50;
    sliderSharpen.preferredSize.width = 212;
    sliderSharpen.alignment = ["center", "top"];

    var labelSharpen = generalPanel.add("statictext", undefined, sliderSharpen.value + "%", {
        name: "labelSharpen"
    });
    labelSharpen.alignment = ["center", "top"];

    sliderSharpen.onChange = function() {
        labelSharpen.text = Math.round(sliderSharpen.value) + "%";
    }

    var textSceneChange = generalPanel.add("statictext", undefined, undefined, {
        name: "textSceneChange"
    });
    textSceneChange.text = "Auto Cut Sensitivity";
    textSceneChange.justify = "center";
    textSceneChange.alignment = ["center", "top"];

    var sliderSceneChange = generalPanel.add("slider", undefined, undefined, undefined, undefined, {
        name: "sliderSceneChange"
    });
    sliderSceneChange.minvalue = 0;
    sliderSceneChange.maxvalue = 100;
    sliderSceneChange.value = 50;
    sliderSceneChange.preferredSize.width = 212;
    sliderSceneChange.alignment = ["center", "top"];

    labelSceneChange = generalPanel.add("statictext", undefined, sliderSceneChange.value + "%", {
        name: "labelSceneChange"
    });
    labelSceneChange.alignment = ["center", "top"];

    sliderSceneChange.onChange = function() {
        labelSceneChange.text = Math.round(sliderSceneChange.value) + "%";
    }

    var textDedupSens = generalPanel.add("statictext", undefined, undefined, {
        name: "textDedupSens"
    });
    textDedupSens.text = "Deduplication Sensitivity";
    textDedupSens.justify = "center";
    textDedupSens.alignment = ["center", "top"];

    var sliderDedupSens = generalPanel.add("slider", undefined, undefined, undefined, undefined, {
        name: "sliderDedupSens"
    });
    sliderDedupSens.minvalue = 0;
    sliderDedupSens.maxvalue = 100;
    sliderDedupSens.value = 50;
    sliderDedupSens.preferredSize.width = 212;
    sliderDedupSens.alignment = ["center", "top"];

    var labelDedupSens = generalPanel.add("statictext", undefined, sliderDedupSens.value + "%", {
        name: "labelDedupSens"
    });
    labelDedupSens.alignment = ["center", "top"];

    sliderDedupSens.onChange = function() {
        labelDedupSens.text = Math.round(sliderDedupSens.value) + "%";
    }

    // GROUP2
    // ======
    var group2 = generalPanel.add("group", undefined, {
        name: "group2"
    });
    group2.orientation = "row";
    group2.alignChildren = ["left", "center"];
    group2.spacing = 0;
    group2.margins = 0;

    var textInterpolationMultiplier = group2.add("statictext", undefined, undefined, {
        name: "textInterpolationMultiplier"
    });
    textInterpolationMultiplier.text = "Interpolation Multiplier";
    textInterpolationMultiplier.preferredSize.width = 172;
    textInterpolationMultiplier.alignment = ["left", "center"];

    var intInterpolate = group2.add('edittext {justify: "center", properties: {name: "intInterpolate"}}');
    intInterpolate.text = "2";
    intInterpolate.preferredSize.width = 40;
    intInterpolate.alignment = ["left", "center"];

    // GROUP3
    // ======
    var group3 = generalPanel.add("group", undefined, {
        name: "group3"
    });
    group3.orientation = "row";
    group3.alignChildren = ["left", "center"];
    group3.spacing = 0;
    group3.margins = 0;

    var textUpscaleMultiplier = group3.add("statictext", undefined, undefined, {
        name: "textUpscaleMultiplier"
    });
    textUpscaleMultiplier.text = "Upscale Multiplier";
    textUpscaleMultiplier.preferredSize.width = 172;

    var intUpscale = group3.add('edittext {justify: "center", properties: {name: "intUpscale"}}');
    intUpscale.text = "2";
    intUpscale.preferredSize.width = 40;
    intUpscale.alignment = ["left", "top"];

    // GROUP4
    // ======
    var group4 = generalPanel.add("group", undefined, {
        name: "group4"
    });
    group4.orientation = "row";
    group4.alignChildren = ["left", "center"];
    group4.spacing = 0;
    group4.margins = 0;

    // PANEL1
    // ======
    var panel1 = settingsWindow.add("panel", undefined, undefined, {
        name: "panel1"
    });
    panel1.text = "Advanced Settings";
    panel1.orientation = "column";
    panel1.alignChildren = ["left", "top"];
    panel1.spacing = 10;
    panel1.margins = 10;

    // GROUP5
    // ======
    var group5 = panel1.add("group", undefined, {
        name: "group5"
    });
    group5.orientation = "row";
    group5.alignChildren = ["left", "center"];
    group5.spacing = 0;
    group5.margins = 0;

    var textUpscaleModel = group5.add("statictext", undefined, undefined, {
        name: "textUpscaleModel"
    });
    textUpscaleModel.text = "Upscale Model";
    textUpscaleModel.preferredSize.width = 103;

    var dropdownModel_array = ["ShuffleCugan", "-", "Compact", "-", "UltraCompact", "-", "SuperUltraCompact", "-", "Cugan", "-", "Cugan-AMD", "-", "SwinIR"];
    var dropdownModel = group5.add("dropdownlist", undefined, undefined, {
        name: "dropdownModel",
        items: dropdownModel_array
    });
    dropdownModel.helpTip = "Choose which model you want to utilize, ordered by speed, read more in INFO";
    dropdownModel.selection = 0;
    dropdownModel.preferredSize.width = 109;

    // GROUP 6
    // ======

    var group6 = panel1.add("group", undefined, {
        name: "group6"
    });

    group6.orientation = "row";
    group6.alignChildren = ["left", "center"];
    group6.spacing = 0;
    group6.margins = 0;

    var textInterpolateModel = group6.add("statictext", undefined, undefined, {
        name: "textInterpolateModel"
    });
    textInterpolateModel.text = "Interpolate Model";
    textInterpolateModel.preferredSize.width = 103;
    textInterpolateModel.helpTip = "Choose which interpolation model you want to utilize, ordered by speed, GFMSS should only really be used on systems with 3080 / 4070 or higher, read more in INFO";

    var dropdownInterpolate = ["Rife_4.14", "-", "Rife_4.14_lite", "-" , "Rife_4.13_lite", "-", "GMFSS"];
    var dropdownInterpolate = group6.add("dropdownlist", undefined, undefined, {
        name: "dropdownInterpolate",
        items: dropdownInterpolate
    });
    dropdownInterpolate.helpTip = "Choose which interpolation model you want to utilize, ordered by speed, GFMSS should only really be used on systems with 3080 / 4070 or higher, read more in INFO";
    dropdownInterpolate.selection = 0;
    dropdownInterpolate.preferredSize.width = 109;

    // GROUP 7
    // ======
    var group7 = panel1.add("group", undefined, {
        name: "group7"
    });
    group7.orientation = "row";
    group7.alignChildren = ["left", "center"];
    group7.spacing = 0;
    group7.margins = 0;

    var cuganDenoiseText = group7.add("statictext", undefined, undefined, {
        name: "cuganDenoiseText"
    });
    cuganDenoiseText.text = "Cugan Denoise";
    cuganDenoiseText.preferredSize.width = 103;

    var dropdownCugan_array = ["No-Denoise", "-", "Conservative", "-", "Denoise1x", "-", "Denoise2x"];
    var dropdownCugan = group7.add("dropdownlist", undefined, undefined, {
        name: "dropdownCugan",
        items: dropdownCugan_array
    });
    dropdownCugan.selection = 0;
    dropdownCugan.preferredSize.width = 109;

    // Create a new group8
    var group8 = panel1.add("group", undefined, {
        name: "group8"
    });

    group8.orientation = "row";
    group8.alignChildren = ["left", "center"];
    group8.spacing = 0;
    group8.margins = 0;

    // Move the encoder to group8
    var textEncoderSelection = group8.add("statictext", undefined, undefined, {
        name: "textEncoderSelection"
    });

    textEncoderSelection.text = "Encoder";
    textEncoderSelection.preferredSize.width = 103;
    textEncoderSelection.helpTip = "Choose which encoder you want to utilize, in no specific order, NVENC for NVidia GPUs and QSV for Intel iGPUs";

    var dropdownEncoder_array = ["X264", "-", "X264_Animation", "-" , "AV1", "-", "NVENC_H264", "-", "NVENC_H265", "-", "QSV_H264", "-", "QSV_H265", "-", "NVENC_AV1"];
    var dropdownEncoder = group8.add("dropdownlist", undefined, undefined, {
        name: "dropdownEncoder",
        items: dropdownEncoder_array
    });

    dropdownEncoder.selection = 0;
    dropdownEncoder.preferredSize.width = 109;

    // GROUP10
    var buttonSettingsClose = settingsWindow.add("button", undefined, undefined, {
        name: "buttonSettingsClose"
    });
    buttonSettingsClose.text = "Close";

    buttonSettingsClose.onClick = function() {
        settingsWindow.hide();
    }

    buttonOutput.onClick = function() {
        var folder = Folder.selectDialog("Select Output folder");
        if (folder != null) {
            outputFolder = folder.fsName;
            app.settings.saveSetting(scriptName, "outputFolder", outputFolder);

            textOutputFolderValue.text = outputFolder;
        }
    };

    buttonFolder.onClick = function() {
        var folder = Folder.selectDialog("Select The Anime Scripter folder");
        if (folder != null) {
            TheAnimeScripterPath = folder.fsName;
            app.settings.saveSetting(scriptName, "TheAnimeScripterPath", TheAnimeScripterPath);

            textTheAnimeScripterFolderValue.text = TheAnimeScripterPath;
        }
    };

    dropdownModel.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdownModel", dropdownModel.selection.index);
    }

    dropdownCugan.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdownCugan", dropdownCugan.selection.index);
    }

    dropdownSwinIr.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdownSwinIr", dropdownSwinIr.selection.index);
    }

    dropdwonSegment.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdwonSegment", dropdwonSegment.selection.index);
    }

    intInterpolate.onChange = function() {
        app.settings.saveSetting(scriptName, "intInterpolate", intInterpolate.text);
    }

    intUpscale.onChange = function() {
        app.settings.saveSetting(scriptName, "intUpscale", intUpscale.text);
    }

    dropdownDedupStrenght.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdownDedupStrenght", dropdownDedupStrenght.selection.index);
    }

    buttonSettings.onClick = function() {
        settingsWindow.show();
    };

    sliderSharpen.onChanging = function() {
        app.settings.saveSetting(scriptName, "sliderSharpen", sliderSharpen.value);
    }

    sliderSceneChange.onChanging = function() {
        app.settings.saveSetting(scriptName, "sliderSceneChange", sliderSceneChange.value);
    }

    buttonSceneChange.onClick = function() {
        sceneChangeValue = 1;
        start_chain();
    }

    buttonDepthMap.onClick = function() {
        depthValue = 1;
        start_chain();
    }

    buttonSegment.onClick = function() {
        segmentValue = 1;
        start_chain();
    }

    buttonMotionBlur.onClick = function() {
        motionBlurValue = 1;
        start_chain();
    }

    dropdownEncoder.onChange = function() {
        app.settings.saveSetting(scriptName, "dropdownEncoder", dropdownEncoder.selection.index);
    }

    buttonStartProcess.onClick = function() {
        if (checkboxDeduplicate.value == false && checkboxUpscale.value == false && checkboxInterpolate.value == false && checkboxSharpen.value == false) {
            alert("Please select at least one of the checkboxes");
            return;
        }

        start_chain();
    }

    function callCommand(command) {
        try {
            if (command) {
                var cmdCommand = 'cmd.exe /c "' + command + '"';
                //alert(cmdCommand)
                system.callSystem(cmdCommand);
            } else {
                throw new Error("Command is undefined");
            }
        } catch (error) {
            alert("Something went wrong trying to process the chain, please contact me on discord");
            return error.toString();
        }
        return null;
    }


    function start_chain() {
        if (((!app.project) || (!app.project.activeItem)) || (app.project.activeItem.selectedLayers.length < 1)) {
            alert("Please select one layer.");
            return;
        }

        if (outputFolder == "undefined" || outputFolder == null) {
            alert("The output folder has not been selected, please go to settings");
            return;
        }

        if (TheAnimeScripterPath == "undefined" || TheAnimeScripterPath == null) {
            alert("The Anime Scripter directory has not been selected, please go to settings");
            return;
        }

        if (app.preferences.getPrefAsLong("Main Pref Section v2", "Pref_SCRIPTING_FILE_NETWORK_SECURITY") != 1) {
            alert("Please tick the \"Allow Scripts to Write Files and Access Network\" checkbox in Scripting & Expressions");
            return app.executeCommand(2359);
        }

        var exeFile = TheAnimeScripterPath + "\\main.exe";

        var exeFilePath = new File(exeFile);
        if (!exeFilePath.exists) {
            alert("Cannot find main.exe, please make sure you have selected the correct folder in settings!");
            return;
        }

        var activeItem = app.project.activeItem;

        var comp = app.project.activeItem;
        var i = 0;
        var selectedLayers = comp.selectedLayers.slice();
        while (i < selectedLayers.length) {
            var layer = selectedLayers[i];
            var activeLayerPath = layer.source.file.fsName;
            var activeLayerName = layer.name;

            var sourceInPoint, sourceOutPoint;

            sourceInPoint = layer.inPoint - layer.startTime;
            sourceOutPoint = layer.outPoint - layer.startTime;

            if (layer.inPoint > sourceInPoint) {
                sourceInPoint = layer.inPoint - layer.startTime;
            }
            if (layer.outPoint < sourceOutPoint) {
                sourceOutPoint = layer.outPoint - layer.startTime;
            }

            var randomNumbers = Math.floor(Math.random() * 10000);
            output_name = outputFolder + "\\" + activeLayerName.replace(/\.[^\.]+$/, '') + "_" + "TAS" + randomNumbers + ".mp4";

            try {
                var attempt = [
                    "cd", "\"" + TheAnimeScripterPath + "\"",
                    "&&",
                    "\"" + exeFile + "\"",
                    "--input", "\"" + activeLayerPath + "\"",
                    "--output", "\"" + output_name + "\"",
                    "--interpolate", checkboxInterpolate.value ? "1" : "0",
                    "--interpolate_factor", intInterpolate.text,
                    "--interpolate_method", dropdownInterpolate.selection.text,
                    "--upscale", checkboxUpscale.value ? "1" : "0",
                    "--upscale_factor", intUpscale.text,
                    "--upscale_method", dropdownModel.selection.text,
                    "--dedup", checkboxDeduplicate.value ? "1" : "0",
                    "--dedup_sens", sliderDedupSens.value,
                    "--half", "1",
                    "--inpoint", sourceInPoint,
                    "--outpoint", sourceOutPoint,
                    "--sharpen", checkboxSharpen.value ? "1" : "0",
                    "--sharpen_sens", sliderSharpen.value,
                    "--segment", segmentValue,
                    "--scenechange", sceneChangeValue,
                    "--depth", depthValue,
                    "--encode_method", dropdownEncoder.selection.text,
                    "--scenechange_sens", 100 - sliderSceneChange.value,
                    "--motion_blur", motionBlurValue,
                ];
                var command = attempt.join(" ");
                callCommand(command);
            } catch (error) {
                alert(error);
            }

            while (true) {
                if (sceneChangeValue == 1) {
                    var sceneChangeLogPath = TheAnimeScripterPath + "\\_internal" + "\\scenechangeresults.txt";
                    var sceneChangeLog = new File(sceneChangeLogPath);
                    sceneChangeLog.open("r");

                    inPoint = layer.inPoint;
                    outPoint = layer.outPoint;

                    while (!sceneChangeLog.eof) {
                        var line = sceneChangeLog.readln();
                        var timestamp = parseFloat(line) + inPoint;
                        var duplicateLayer = layer.duplicate();
                        layer.outPoint = timestamp;
                        duplicateLayer.inPoint = timestamp;

                        layer = duplicateLayer;
                    }
                    sceneChangeLog.close();
                    break;
                } else {
                    // Increased Max Attempts, just in case
                    var maxAttempts = 10;
                    for (var attempt = 0; attempt < maxAttempts; attempt++) {
                        $.sleep(1000); // Sleeping for a second, metadata is not always written instantly
                        try {
                            var importOptions = new ImportOptions(File(output_name));
                            var importedFile = app.project.importFile(importOptions);
                            var inputLayer = comp.layers.add(importedFile);
                            inputLayer.startTime = layer.inPoint;
                            inputLayer.moveBefore(layer);
                            if (checkboxUpscale.value == true) {
                                var compWidth = comp.width;
                                var compHeight = comp.height;
                                var layerWidth = inputLayer.source.width;
                                var layerHeight = inputLayer.source.height;
                                var scaleX = (compWidth / layerWidth) * 100;
                                var scaleY = (compHeight / layerHeight) * 100;
                                inputLayer.property("Scale").setValue([scaleX, scaleY, 100]);
                            } else {
                                var layerWidth = layer.width;
                                var layerHeight = layer.height;
                                var scaleX = (layerWidth / inputLayer.source.width) * 100;
                                var scaleY = (layerHeight / inputLayer.source.height) * 100;
                                inputLayer.property("Scale").setValue([scaleX, scaleY, 100]);
                            }
                            break;
                        } catch (error) {
                            if (attempt == maxAttempts - 1) {
                                alert("Failed to import file after " + maxAttempts + " attempts: " + output_name);
                                alert("Error: " + error.toString());
                            }
                        }
                    }
                    break;
                }
            }
            i++;
        }
        sceneChangeValue = 0;
        segmentValue = 0;
        depthValue = 0;
        motionBlurValue = 0;
    }
    if (TheAnimeScripter instanceof Window) TheAnimeScripter.show();
    return TheAnimeScripter;
}());