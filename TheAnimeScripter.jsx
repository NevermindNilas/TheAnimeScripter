var panelGlobal = this;
var TheAnimeScripter = (function() {

    var scriptName = "TheAnimeScripter";
    var scriptVersion = "1.2.0";

    /*
    scriptAuthor = "Nilas";
    scriptURL = "https://github.com/NevermindNilas/TheAnimeScripter"
    discordServer = "https://discord.gg/CdRD9GwS8J"
    */

    // Default Values for the settings
    var outputFolder = app.settings.haveSetting(scriptName, "outputFolder") ? app.settings.getSetting(scriptName, "outputFolder") : "undefined";
    var theAnimeScripterPath = app.settings.haveSetting(scriptName, "theAnimeScripterPath") ? app.settings.getSetting(scriptName, "theAnimeScripterPath") : "undefined";
    var segmentValue = 0;
    var sceneChangeValue = 0;
    var depthValue = 0;
    var customModelPath = "undefined";

    var exeFile = theAnimeScripterPath + "\\main.exe";
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

    var group0 = panelChain.add("group", undefined, {
        name: "group0"
    });

    group0.orientation = "row";
    group0.alignChildren = ["left", "center"];
    group0.spacing = 10;
    group0.margins = 0;

    var checkboxResize = group0.add("checkbox", undefined, "Resize", {
        name: "checkboxResize"
    });
    checkboxResize.alignment = ["left", "center"];
    checkboxResize.helpTip = "Resize by a desired factor before further processing, meant as an substitute for upscaling on lower end GPUs";

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

    buttonGetVideo = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonGetVideo"
    });

    buttonGetVideo.text = "Get Video";
    buttonGetVideo.preferredSize.width = 105;
    buttonGetVideo.alignment = ["center", "top"];
    buttonGetVideo.helpTip = "Get Video from Youtube using YT-DLP";

    textGetVideo = panelPostProcess.add("edittext", undefined, undefined, {
        name: "textGetVideo"
    });

    textGetVideo.text = " Add Youtube URL";
    textGetVideo.preferredSize.width = 105;
    textGetVideo.alignment = ["center", "top"];


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

    var buttonPreRender = panelMore.add("button", undefined, "Pre-Render", {
        name: "buttonPreRender"
    });

    buttonPreRender.text = "Pre-Render";
    buttonPreRender.alignment = ["left", "center"];
    buttonPreRender.helpTip = "Pre-Render the video before further processing, useful for already modified videos or multi layered compositions";
    buttonPreRender.preferredSize.width = 105;

    var buttonSettings = panelMore.add("button", undefined, undefined, {
        name: "buttonSettings"
    });
    buttonSettings.text = "Settings";
    buttonSettings.preferredSize.width = 105;
    buttonSettings.alignment = ["center", "top"];

    var textScriptVersion = TheAnimeScripter.add("statictext", undefined, undefined, {
        name: "textScriptVersion"
    });

    textScriptVersion.text = "Script Version: " + scriptVersion;

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
    
    var labelValues = {};
    function createSlider(panel, text, name, defaultValue) {
        var group = panel.add("group", undefined, { name: "group" + name });
        group.orientation = "row";
        group.alignChildren = ["fill", "center"];

        var staticText = group.add("statictext", undefined, text, { name: "text" + name });
        staticText.justify = "center";
        staticText.alignment = ["left", "center"];

        var filler = group.add("statictext", undefined, "", { name: "filler" + name });
        filler.alignment = ["fill", "center"];

        var label = group.add("statictext", undefined, defaultValue + "%", { name: "label" + name });
        label.alignment = ["right", "center"];

        var slider = panel.add("slider", undefined, defaultValue, 0, 100, { name: "slider" + name });
        slider.preferredSize.width = 212;
        slider.alignment = ["center", "top"];

        labelValues[name] = defaultValue;

        slider.onChange = function() {
            var value = Math.round(slider.value);
            label.text = value + "%";
            labelValues[name] = value;
        }
    }

    createSlider(generalPanel, "Sharpenening Sensitivity", "Sharpen", 50);
    createSlider(generalPanel, "Auto Cut Sensitivity", "SceneChange", 50);
    createSlider(generalPanel, "Deduplication Sensitivity", "DedupSens", 50);

    var sharpenSensValue = function() { return labelValues["Sharpen"]; };
    var sceneChangeSensValue = function() { return labelValues["SceneChange"]; };
    var dedupSensValue = function() { return labelValues["DedupSens"]; };

    var group4 = generalPanel.add("group", undefined, {
        name: "group4"
    });

    group4.orientation = "row";
    group4.alignChildren = ["left", "center"];
    group4.spacing = 45;
    group4.margins = 0;

    var checkboxEnsemble = group4.add("checkbox", undefined, "Rife Ensemble", {
        name: "checkboxEnsemble"
    });

    checkboxEnsemble.alignment = ["left", "center"];
    checkboxEnsemble.helpTip = "Turn on ensemble for Rife, higher quality outputs for a slight tax in performance";

    var checkboxYTDLPQuality = group4.add("checkbox", undefined, "YT-DLP 4K", {
        name: "checkboxYTDLPQuality"
    });

    checkboxYTDLPQuality.alignment = ["left", "center"];
    checkboxYTDLPQuality.helpTip = "Turn on higher quality download for YT-DLP, will download the highest quality available from yt (4k/8k) and then re-encode the video to the desired encoder, turn off for 1920x1080p only downloads";
    
    var fieldValues = {}
    function createMultiplierField(panel, text, name, defaultValue) {
        var group = panel.add("group", undefined, { name: "group" + name });
        group.orientation = "row";
        group.alignChildren = ["fill", "center"];
        group.spacing = 0;
        group.margins = 0;

        var staticText = group.add("statictext", undefined, undefined, { name: "text" + name });
        staticText.text = text;
        staticText.preferredSize.width = 172;
        staticText.alignment = ["left", "center"];

        var filler = group.add("statictext", undefined, "", { name: "filler" + name });
        filler.alignment = ["fill", "center"];

        var editText = group.add('edittext {justify: "center", properties: {name: "' + name + '"}}');
        editText.text = defaultValue;
        editText.preferredSize.width = 40;
        editText.alignment = ["right", "center"];

        editText.onChange = function() {
            fieldValues[name] = editText.text;
        }

        fieldValues[name] = defaultValue;

        return editText;
    }

    createMultiplierField(generalPanel, "Resize Multiplier", "Resize", "2");
    createMultiplierField(generalPanel, "Interpolation Multiplier", "Interpolate", "2");
    createMultiplierField(generalPanel, "Upscale Multiplier", "Upscale", "2");
    createMultiplierField(generalPanel, "Number of Threads", "Threads", "1")

    var resizeValue = function() { return fieldValues["Resize"]; };
    var interpolateValue = function() { return fieldValues["Interpolate"]; };
    var upscaleValue = function() { return fieldValues["Upscale"]; };
    var threadsValue = function() { return fieldValues["Threads"]; };
    
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

    var dropdownValues = {}

    function createDropdownField(panel, text, name, dropdownArray, helpTip) {
        var group = panel.add("group", undefined, { name: "group" + name });
        group.orientation = "row";
        group.alignChildren = ["left", "center"];
        group.spacing = 0;
        group.margins = 0;

        var staticText = group.add("statictext", undefined, undefined, { name: "text" + name });
        staticText.text = text;
        staticText.preferredSize.width = 103;

        var dropdown = group.add("dropdownlist", undefined, undefined, {
            name: "dropdown" + name,
            items: dropdownArray
        });
        dropdown.helpTip = helpTip;
        dropdown.selection = 0;
        dropdown.preferredSize.width = 109;

        dropdown.onChange = function() {
            dropdownValues[name] = dropdown.selection.text;
        }

        dropdownValues[name] = dropdown.selection.text;

        return dropdown;
    }

    createDropdownField(panel1, "Upscale Model", "Model", ["ShuffleCugan", "-", "Compact", "-", "UltraCompact", "-", "SuperUltraCompact", "-", "Cugan", "-", "Cugan-NCNN", "-", "Span", "-", "Span-NCNN", "-", "SwinIR", "-", "OmniSR"], "Choose which model you want to utilize, read more in INFO, for AMD users choose NCNN models");
    createDropdownField(panel1, "Interpolate Model", "Interpolate", ["Rife4.14", "-", "Rife4.14-Lite", "-" , "Rife4.13-Lite", "-", "Rife4.6", "-", "Rife4.14-NCNN", "-", "Rife4.14-Lite-NCNN", "-", "Rife4.13-Lite-NCNN", "-", "Rife4.6-NCNN", "-", "GMFSS"], "Choose which interpolation model you want to utilize, ordered by speed, GFMSS should only really be used on systems with 3080 / 4070 or higher, read more in INFO");
    createDropdownField(panel1, "Cugan Denoise", "Cugan", ["No-Denoise", "-", "Conservative", "-", "Denoise1x", "-", "Denoise2x"]);
    createDropdownField(panel1, "Depth Model", "Depth", ["Small", "-", "Base", "-", "Large"], "Choose which depth map model you want to utilize, ordered by speed, read more in INFO");
    createDropdownField(panel1, "Encoder", "Encoder", ["X264", "-", "X264_Animation", "-" , "X265", "-", "AV1", "-", "NVENC_H264", "-", "NVENC_H265", "-", "NVENC_AV1", "-", "QSV_H264", "-", "QSV_H265", "-", "H264_AMF", "-", "HEVC_AMF"], "Choose which encoder you want to utilize, in no specific order, NVENC for NVidia GPUs and QSV for Intel iGPUs");
    createDropdownField(panel1, "Resize Method", "Resize", ["Fast_Bilinear", "-", "Bilinear", "-", "Bicubic", "-", "Experimental", "-", "Neighbor", "-", "Area", "-", "Bicublin", "-", "Gauss", "-", "Sinc", "-", "Lanczos", "-", "Spline", "-",  "Spline16", "-", "Spline36"], "Choose which resize method you want to utilize, For upscaling I would suggest Lanczos or Spline, for downscaling I would suggest Area or Bicubic");

    var upscaleModel = function() { return dropdownValues["Model"]; };
    var interpolateModel = function() { return dropdownValues["Interpolate"]; };
    var cuganDenoise = function() { return dropdownValues["Cugan"]; };
    var depthModel = function() { return dropdownValues["Depth"]; };
    var encoderMethod = function() { return dropdownValues["Encoder"]; };
    var resizeMethod = function() { return dropdownValues["Resize"]; };

    var panelCustomSettings = settingsWindow.add("panel", undefined, undefined, {
        name: "panelCustomSettings"
    });

    panelCustomSettings.text = "Custom Settings";
    panelCustomSettings.orientation = "column";
    panelCustomSettings.alignChildren = ["left", "top"];

    var buttonCustomModel = panelCustomSettings.add("button", [0,0,202,30], undefined, {
        name: "buttonCustomModel"
    });

    buttonCustomModel.text = "Custom Upscaling Model";
    buttonCustomModel.preferredSize.width = 105;
    buttonCustomModel.alignment = ["center", "top"];
    buttonCustomModel.helpTip = "Add a custom model to utilize, can be either .pth or .onnx, this will override the custom model of your choice in the dropdown, it has to be one of the supported architectures ( Shufflecugan, Cugan ... ) and it also relies on the upscale multiplier to match the model";

    var labelCustomEncoder = panelCustomSettings.add("statictext", undefined, undefined, {
        name: "labelCustomEncoder"
    });
    
    labelCustomEncoder.text = "Custom FFMPEG Encoding Parameteres";
    labelCustomEncoder.alignment = ["center", "top"];

    var textCustomEncoder = panelCustomSettings.add("edittext", [0,0,202,25], undefined, {
        name: "textCustomEncoder"
    });

    textCustomEncoder.text = "";
    textCustomEncoder.preferredSize.width = 105;
    textCustomEncoder.alignment = ["center", "top"];
    textCustomEncoder.helpTip = "Add custom FFMPEG encoding parameters, this will override the encoding method of your choice in the dropdown, you need to declare every new option including codec, video and filters that you might like to use, for example: -c:v libx264 -crf 23 -preset veryfast -tune animation -vf \"scale=1920:1080\"";
    
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
        alert("Output folder set to: " + outputFolder);
    };

    buttonFolder.onClick = function() {
        var folder = Folder.selectDialog("Select The Anime Scripter folder");
        if (folder != null) {
            theAnimeScripterPath = folder.fsName;
            app.settings.saveSetting(scriptName, "theAnimeScripterPath", theAnimeScripterPath);

            textTheAnimeScripterFolderValue.text = theAnimeScripterPath;
        }
        alert("The Anime Scripter folder set to: " + theAnimeScripterPath);
    };

    buttonSettings.onClick = function() {
        settingsWindow.show();
    };

    buttonSceneChange.onClick = function() {
        sceneChangeValue = 1;
        startChain();
    }

    buttonDepthMap.onClick = function() {
        depthValue = 1;
        startChain();
    }

    buttonSegment.onClick = function() {
        segmentValue = 1;
        startChain();
    }

    buttonStartProcess.onClick = function() {
        if (checkboxDeduplicate.value == false && checkboxUpscale.value == false && checkboxInterpolate.value == false && checkboxSharpen.value == false) {
            if (checkboxResize.value == false) {
                alert("Please select at least one of the checkboxes");
                return;
            }
        }

        startChain();
    }

    buttonGetVideo.onClick = function(){
        if (textGetVideo.text == "Add Youtube URL") {
            alert("Please add a Youtube URL");
            return;
        }

        startDownload();
    }

    buttonPreRender.onClick = function() {
        preRender();
    }

    buttonCustomModel.onClick = function() {
        var customModel = File.openDialog("Select a Custom Model", "*.pth;*.onnx", false);
        if (customModel != null) {
            customModelPath = customModel.fsName;
            customModelName = customModel.name;
            alert("Custom Model set to: " + customModelName);
        }
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

    function preRender() {
        if (typeof outputFolder === 'undefined') {
            alert("The output folder has not been selected, please go to settings");
            return;
        }

        var comp = app.project.activeItem;
        var selectedLayers = comp.selectedLayers;

        var maxEndTime = 0;
        for (var i = 0; i < selectedLayers.length; i++) {
            var layer = selectedLayers[i];
            var endTime = layer.startTime + layer.outPoint;
            if (endTime > maxEndTime) {
                maxEndTime = endTime;
            }
        }

        var newComp = app.project.items.addComp('New Composition', comp.width, comp.height, comp.pixelAspect, maxEndTime, comp.frameRate);

        for (var i = 0; i < selectedLayers.length; i++) {
            var layer = selectedLayers[i];
            var newLayer = newComp.layers.add(layer.source);
            newLayer.startTime = layer.startTime;
            newLayer.inPoint = layer.inPoint;
            newLayer.outPoint = layer.outPoint;
            newLayer.enabled = layer.enabled;
            newLayer.solo = layer.solo;
            newLayer.shy = layer.shy;
            newLayer.locked = layer.locked;
            newLayer.name = layer.name;

            // Copy transformations
            newLayer.position.setValue(layer.position.value);
            newLayer.scale.setValue(layer.scale.value);
            newLayer.rotation.setValue(layer.rotation.value);
            newLayer.opacity.setValue(layer.opacity.value);

            for (var j = 1; j <= layer.effect.numProperties; j++) {
                var effect = layer.effect(j);
                var newEffect = newLayer.effect.addProperty(effect.matchName);
                for (var k = 1; k <= effect.numProperties; k++) {
                    newEffect.property(k).setValue(effect.property(k).value);
                }
            }
        }

        var renderQueue = app.project.renderQueue;
        var render = renderQueue.items.add(newComp);
        var outputModule = render.outputModule(1);

        randomNumbers = Math.floor(Math.random() * 1000);
        var outputFileExtension = (parseFloat(app.version) >= 23.0) ? ".mp4" : ".mov";
        var outputName = outputFolder + "/TAS_" + randomNumbers + outputFileExtension;
        outputModule.file = new File(outputName);

        renderQueue.render();
        var importedFile = app.project.importFile(new ImportOptions(new File(outputName)));
        var newLayer = comp.layers.add(importedFile);

        newComp.remove();
    }

    function startChain() {
        if (((!app.project) || (!app.project.activeItem)) || (app.project.activeItem.selectedLayers.length < 1)) {
            alert("Please select one layer.");
            return;
        }

        if (outputFolder == "undefined" || outputFolder == null) {
            alert("The output folder has not been selected, please go to settings");
            return;
        }

        if (theAnimeScripterPath == "undefined" || theAnimeScripterPath == null) {
            alert("The Anime Scripter directory has not been selected, please go to settings");
            return;
        }

        if (app.preferences.getPrefAsLong("Main Pref Section v2", "Pref_SCRIPTING_FILE_NETWORK_SECURITY") != 1) {
            alert("Please tick the \"Allow Scripts to Write Files and Access Network\" checkbox in Scripting & Expressions");
            return app.executeCommand(2359);
        }

        var exeFile = theAnimeScripterPath + "\\main.exe";
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

            if (segmentValue == 1) {
                var outputFileExtension = ".mov";
            } else {
                var outputFileExtension = ".mp4";
            }

            randomNumbers = Math.floor(Math.random() * 1000);
            output_name = outputFolder + "\\" + activeLayerName.replace(/\.[^\.]+$/, '') + "-" + randomNumbers + outputFileExtension;

            try {
                var attempt = [
                    "cd", "\"" + theAnimeScripterPath + "\"",
                    "&&",
                    "\"" + exeFile + "\"",
                    "--input", "\"" + activeLayerPath + "\"",
                    "--output", "\"" + output_name + "\"",
                    "--interpolate", checkboxInterpolate.value ? "1" : "0",
                    "--interpolate_factor", interpolateValue(),
                    "--interpolate_method", interpolateModel().toLowerCase(),
                    "--upscale", checkboxUpscale.value ? "1" : "0",
                    "--upscale_factor", upscaleValue(),
                    "--upscale_method", upscaleModel().toLowerCase(),
                    "--dedup", checkboxDeduplicate.value ? "1" : "0",
                    "--dedup_sens", dedupSensValue(),
                    "--half", "1",
                    "--inpoint", sourceInPoint,
                    "--outpoint", sourceOutPoint,
                    "--sharpen", checkboxSharpen.value ? "1" : "0",
                    "--sharpen_sens", sharpenSensValue(),
                    "--segment", segmentValue,
                    "--scenechange", sceneChangeValue,
                    "--depth", depthValue,
                    "--depth_method", depthModel().toLowerCase(),
                    "--encode_method", encoderMethod().toLowerCase(),
                    "--scenechange_sens", sceneChangeSensValue(),
                    "--ensemble", checkboxEnsemble.value ? "1" : "0",
                    "--resize", checkboxResize.value ? "1" : "0",
                    "--resize_method", resizeMethod().toLowerCase(),
                    "--resize_factor", resizeValue(),
                    "--nt", threadsValue(),
		    "--buffer_limit", 200,
                ];

                if (customModelPath && customModelPath !== "undefined") {
                    attempt.push("--custom_model", customModelPath);
                }

                var command = attempt.join(" ");
                callCommand(command);
            } catch (error) {
                alert(error);
            }

            while (true) {
                if (sceneChangeValue == 1) {
                    var sceneChangeLogPath = theAnimeScripterPath + "\\scenechangeresults.txt";
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
                    var maxAttempts = 3;
                    for (var attempt = 0; attempt < maxAttempts; attempt++) {
                        $.sleep(1000); // Sleeping for a second, metadata is not always written instantly
                        try {
                            var importOptions = new ImportOptions(File(output_name));
                            var importedFile = app.project.importFile(importOptions);
                            var inputLayer = comp.layers.add(importedFile);
                            inputLayer.startTime = layer.inPoint;
                            inputLayer.moveBefore(layer);
                            if (checkboxUpscale.value == true || checkboxResize.value == true) {
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
    }

    function startDownload() {
        if (outputFolder == "undefined" || outputFolder == null) {
            alert("The output folder has not been selected, please go to settings");
            return;
        }

        if (theAnimeScripterPath == "undefined" || theAnimeScripterPath == null) {
            alert("The Anime Scripter directory has not been selected, please go to settings");
            return;
        }

        if (app.preferences.getPrefAsLong("Main Pref Section v2", "Pref_SCRIPTING_FILE_NETWORK_SECURITY") != 1) {
            alert("Please tick the \"Allow Scripts to Write Files and Access Network\" checkbox in Scripting & Expressions");
            return app.executeCommand(2359);
        }

        var exeFile = theAnimeScripterPath + "\\main.exe";

        var exeFilePath = new File(exeFile);
        if (!exeFilePath.exists) {
            alert("Cannot find main.exe, please make sure you have selected the correct folder in settings!");
            return;
        }

        var randomNumbers = Math.floor(Math.random() * 10000);
        var outputName = outputFolder + "\\" + "TAS" + "_YTDLP_" + randomNumbers + ".mp4";

        try {
            var attempt = [
                "cd", "\"" + theAnimeScripterPath + "\"",
                "&&",
                "\"" + exeFile + "\"",
                "--output", "\"" + outputName + "\"",
                "--ytdlp", "\"" + textGetVideo.text + "\"",
                "--ytdlp_quality", checkboxYTDLPQuality.value ? "1" : "0",
            ];
            var command = attempt.join(" ");
            callCommand(command);
        } catch (error) {
            alert(error);
        }
        
        try{
            var newImportOptions = new ImportOptions(File(outputName));
            newImportOptions.importAs = ImportAsType.FOOTAGE;
            
            var comp = app.project.activeItem;
            if (!(comp instanceof CompItem)) {
                alert("No composition is active. Please select a composition and try again.");
                return;
            }
            
            var footageItem = app.project.importFile(newImportOptions);
            var inputLayer = comp.layers.add(footageItem);
        } catch (error) {
            alert(error);
        }
    }

    if (TheAnimeScripter instanceof Window) TheAnimeScripter.show();
    return TheAnimeScripter;
}());