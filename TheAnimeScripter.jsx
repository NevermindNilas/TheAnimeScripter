var panelGlobal = this;
var TheAnimeScripter = (function () {

    var scriptName = "TheAnimeScripter";
    var scriptVersion = "v1.8.9";

    // Default Values for the settings
    var outputFolder = app.settings.haveSetting(scriptName, "outputFolder") ? app.settings.getSetting(scriptName, "outputFolder") : "undefined";
    var theAnimeScripterPath = app.settings.haveSetting(scriptName, "theAnimeScripterPath") ? app.settings.getSetting(scriptName, "theAnimeScripterPath") : "undefined";
    var segmentValue = 0;
    var autoClipValue = 0;
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

    var checkboxValues = {};

    function createCheckboxField(panel, text, name, helpTip) {
        var group = panel.add("group", undefined, {
            name: "group" + name
        });
        group.orientation = "row";
        group.alignChildren = ["left", "top"];
        group.spacing = 0;
        group.margins = 0;

        var checkbox = group.add("checkbox", undefined, undefined, {
            name: "checkbox" + name
        });
        checkbox.helpTip = helpTip;
        var savedValue = app.settings.haveSetting(scriptName, name) ? (app.settings.getSetting(scriptName, name) === "true") : false;
        checkbox.value = savedValue;

        var staticText = group.add("statictext", undefined, undefined, {
            name: "text" + name
        });
        staticText.text = text;

        checkbox.onClick = function () {
            checkboxValues[name] = checkbox.value;
            app.settings.saveSetting(scriptName, name, checkbox.value.toString());
        }

        checkboxValues[name] = savedValue;

        return checkbox;
    }

    createCheckboxField(panelChain, "Resize", "checkboxResize", "Resize by a desired factor before further processing, meant as an substitute for upscaling on lower end GPUs");
    createCheckboxField(panelChain, "Deduplicate", "checkboxDeduplicate", "Deduplicate the video using a desired method, useful for removing duplicate frames and reducing file size");
    createCheckboxField(panelChain, "Denoise", "checkboxDenoise", "Denoise using a desired model");
    createCheckboxField(panelChain, "Upscale", "checkboxUpscale", "Upscale using a desired model and factor");
    createCheckboxField(panelChain, "Interpolate", "checkboxInterpolate", "Interpolate using a desired model and factor");
    createCheckboxField(panelChain, "Sharpen", "checkboxSharpen", "Sharpen the video using a desired factor");

    var checkboxResizeValue = function () {
        return checkboxValues["checkboxResize"];
    };
    var checkboxDeduplicateValue = function () {
        return checkboxValues["checkboxDeduplicate"];
    };
    var checkboxDenoiseValue = function () {
        return checkboxValues["checkboxDenoise"];
    };
    var checkboxUpscaleValue = function () {
        return checkboxValues["checkboxUpscale"];
    };
    var checkboxInterpolateValue = function () {
        return checkboxValues["checkboxInterpolate"];
    };
    var checkboxSharpenValue = function () {
        return checkboxValues["checkboxSharpen"];
    };

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

    var buttonAutoClip = panelPostProcess.add("button", undefined, undefined, {
        name: "buttonAutoClip"
    });

    buttonAutoClip.text = "Auto Cut";
    buttonAutoClip.preferredSize.width = 105;
    buttonAutoClip.alignment = ["center", "top"];

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

    var buttonCheckForUpdate = panelMore.add("button", undefined, undefined, {
        name: "buttonCheckForUpdate"
    });

    buttonCheckForUpdate.text = "Check for Update";
    buttonCheckForUpdate.preferredSize.width = 105;
    buttonCheckForUpdate.alignment = ["center", "top"];

    var textScriptVersion = panelMore.add("statictext", undefined, undefined, {
        name: "textScriptVersion"
    });

    textScriptVersion.text = "Script Version: " + scriptVersion;

    TheAnimeScripter.layout.layout(true);
    TheAnimeScripter.layout.resize();
    TheAnimeScripter.onResizing = TheAnimeScripter.onResize = function () {
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
    buttonOutput.preferredSize.width = 100;
    buttonOutput.helpTip = "Set it to wherever you want the output to be saved.";

    var group2 = panelOnFirstRun.add("group", undefined, {
        name: "group2"
    });

    var buttonCustomModel = group2.add("button", undefined, undefined, {
        name: "buttonCustomModel"
    });

    buttonCustomModel.text = "Custom Upscaler";
    buttonCustomModel.preferredSize.width = 100;
    buttonCustomModel.alignment = ["center", "top"];
    buttonCustomModel.helpTip = "Add a custom model to utilize, can be either .pth or .onnx, this will override the custom model of your choice in the dropdown, it has to be one of the supported architectures ( Shufflecugan, Cugan ... ) and it also relies on the upscale multiplier to match the model";

    var buttonOfflineMode = group2.add("button", undefined, undefined, {
        name: "buttonOfflineMode"
    });

    buttonOfflineMode.text = "Offline Mode";
    buttonOfflineMode.preferredSize.width = 100;
    buttonOfflineMode.alignment = ["center", "top"];
    buttonOfflineMode.helpTip = "This will download all available models and place them in the correct directory, it will take a while and is not recommended for users with slow internet connections, but it will make the script work 100% offline with the exception of Youtube Downloads and Depth Maps";

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
        var savedValue = app.settings.haveSetting(scriptName, name) ? app.settings.getSetting(scriptName, name) : defaultValue;
        var group = panel.add("group", undefined, {
            name: "group" + name
        });
        group.orientation = "column";
        group.alignChildren = ["fill", "center"];

        var labelGroup = group.add("group", undefined, {
            name: "labelGroup" + name
        });
        labelGroup.orientation = "row";
        labelGroup.alignChildren = ["left", "center"];

        var staticText = labelGroup.add("statictext", undefined, text, {
            name: "text" + name
        });
        staticText.justify = "center";
        staticText.alignment = ["left", "center"];

        var label = labelGroup.add("statictext", undefined, savedValue + "%", { // Append "%" to the initial value
            name: "label" + name
        });
        label.alignment = ["right", "center"];

        var slider = group.add("slider", undefined, savedValue, 0, 100, {
            name: "slider" + name
        });
        slider.preferredSize.width = 212;
        slider.alignment = ["center", "top"];

        labelValues[name] = savedValue;

        slider.onChange = function () {
            var value = Math.round(slider.value);
            label.text = value + "%"; // Append "%" to the value set by the slider
            labelValues[name] = value;
            app.settings.saveSetting(scriptName, name, value);
        }
    }

    createSlider(generalPanel, "Sharpenening Sensitivity", "SharpenSens", 50);
    createSlider(generalPanel, "Deduplication Sensitivity", "DedupSens", 50);
    createSlider(generalPanel, "Auto Cut Sensitivity", "AutoClipSens", 50);

    var sharpenSensValue = function () {
        return labelValues["SharpenSens"];
    };
    var sceneChangeSensValue = function () {
        return labelValues["AutoClipSens"];
    };
    var dedupSensValue = function () {
        return labelValues["DedupSens"];
    };


    var fieldValues = {}
    function createMultiplierField(panel, text, name, defaultValue) {
        var group = panel.add("group", undefined, {
            name: "group" + name
        });
        group.orientation = "row";
        group.alignChildren = ["fill", "center"];
        group.spacing = 0;
        group.margins = 0;

        var staticText = group.add("statictext", undefined, undefined, {
            name: "text" + name
        });
        staticText.text = text;
        staticText.preferredSize.width = 172;
        staticText.alignment = ["left", "center"];

        var filler = group.add("statictext", undefined, "", {
            name: "filler" + name
        });
        filler.alignment = ["fill", "center"];

        var savedValue = app.settings.haveSetting(scriptName, name) ? app.settings.getSetting(scriptName, name) : defaultValue;
        var editText = group.add('edittext {justify: "center", properties: {name: "' + name + '"}}');
        editText.text = savedValue;
        editText.preferredSize.width = 40;
        editText.alignment = ["right", "center"];

        editText.onChange = function () {
            fieldValues[name] = editText.text;
            app.settings.saveSetting(scriptName, name, editText.text);
        }

        fieldValues[name] = savedValue;

        return editText;
    }

    createMultiplierField(generalPanel, "Resize Multiplier", "ResizeMultiplier", "2");
    createMultiplierField(generalPanel, "Interpolation Multiplier", "InterpolateMultiplier", "2");
    createMultiplierField(generalPanel, "Upscale Multiplier", "UpscaleMultiplier", "2");
    createMultiplierField(generalPanel, "Number of Threads", "ThreadsMultiplier", "1"); // Ik  this is not a multiplier but it fits the theme

    var resizeValue = function () {
        return fieldValues["ResizeMultiplier"];
    };
    var interpolateValue = function () {
        return fieldValues["InterpolateMultiplier"];
    };
    var upscaleValue = function () {
        return fieldValues["UpscaleMultiplier"];
    };
    var threadsValue = function () {
        return fieldValues["ThreadsMultiplier"];
    };

    createCheckboxField(generalPanel, "Rife Ensemble", "checkboxEnsemble", "Use Rife Ensemble to interpolate frames, this will increase the quality of the video but also the processing time");

    var checkboxEnsembleValue = function () {
        return checkboxValues["checkboxEnsemble"];
    }


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
        var group = panel.add("group", undefined, {
            name: "group" + name
        });
        group.orientation = "row";
        group.alignChildren = ["left", "center"];
        group.spacing = 0;
        group.margins = 0;

        var staticText = group.add("statictext", undefined, undefined, {
            name: "text" + name
        });
        staticText.text = text;
        staticText.preferredSize.width = 103;

        var dropdown = group.add("dropdownlist", undefined, undefined, {
            name: "dropdown" + name,
            items: dropdownArray
        });
        dropdown.helpTip = helpTip;
        var savedValue = app.settings.haveSetting(scriptName, name) ? String(app.settings.getSetting(scriptName, name)) : dropdownArray[0];
        var index = indexOf(dropdownArray, savedValue);
        dropdown.selection = index !== -1 ? dropdown.items[index] : dropdown.items[0];
        dropdown.preferredSize.width = 109;
        dropdown.preferredSize.height = 5;

        dropdown.onChange = function () {
            dropdownValues[name] = dropdown.selection.text;
            app.settings.saveSetting(scriptName, name, dropdown.selection.text); // Save the setting when it changes
        }

        dropdownValues[name] = savedValue;

        return dropdown;
    }

    function indexOf(arr, item) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i] === item) {
                return i;
            }
        }
        return -1;
    }

    createDropdownField(panel1, "Upscale Model", "Model", [
        "ShuffleCugan",
        "Compact",
        "UltraCompact",
        "SuperUltraCompact",
        "Span",
        "ShuffleCugan-TensorRT",
        "Span-TensorRT",
        "Compact-TensorRT",
        "UltraCompact-TensorRT",
        "SuperUltraCompact-TensorRT",
        "Compact-DirectML",
        "UltraCompact-DirectML",
        "SuperUltraCompact-DirectML",
        "Span-DirectML",
        "ShuffleCugan-NCNN",
        "Span-NCNN",
    ], "Choose which model you want to utilize, read more in INFO, for AMD users choose DirectML or NCNN models");
    createDropdownField(panel1, "Interpolate Model", "Interpolate", ["Rife4.18","Rife4.17", "Rife4.17-Lite", "Rife4.16-Lite", "Rife4.15", "Rife4.15-Lite", "Rife4.6", "Rife4.18-TensorRT", "Rife4.17-TensorRT", "Rife4.15-TensorRT", "Rife4.15-Lite-TensorRT", "Rife4.6-TensorRT", "Rife4.16-Lite-NCNN", "Rife4.15-NCNN", "Rife4.15-Lite-NCNN", "Rife4.6-NCNN", "GMFSS"], "Choose which interpolation model you want to utilize, read more in readme.txt, for AMD users choose DirectML or NCNN models, TensorRT is for NVIDIA RTX 2000 and above GPUs");
    createDropdownField(panel1, "Depth Model", "Depth", ["Small_V2", "Base_V2", "Large_V2", "Small_V2-TensorRT", "Base_V2-TensorRT", "Large_V2-TensorRT"], "Choose which depth map model you want to utilize, ordered by speed, read more in INFO");
    createDropdownField(panel1, "Encoder", "Encoder", ["X264", "X264_10Bit", "X264_Animation", "X264_Animation_10Bit", "X265", "X265_10Bit", "NVENC_H264", "NVENC_H265", "NVENC_H265_10Bit", "NVENC_AV1", "QSV_H264", "QSV_H265", "QSV_H265_10Bit", "H264_AMF", "HEVC_AMF", "HEVC_AMF_10Bit", "AV1"], "Choose which encoder you want to utilize, in no specific order, NVENC for NVidia GPUs, AMF for AMD GPUs and QSV for Intel iGPUs");
    createDropdownField(panel1, "Resize Method", "Resize", ["Fast_Bilinear", "Bilinear", "Bicubic", "Experimental", "Neighbor", "Area", "Bicublin", "Gauss", "Sinc", "Lanczos", "Spline", "Spline16", "Spline36"], "Choose which resize method you want to utilize, For upscaling I would suggest Lanczos or Spline, for downscaling I would suggest Area or Bicubic");
    createDropdownField(panel1, "Dedup Method", "Dedup", ["SSIM", "MSE", "SSIM-CUDA"], "Choose which deduplication method you want to utilize, SSIM-CUDA is for NVIDIA Only whilst the rest will work on any system.");
    createDropdownField(panel1, "Denoise Method", "Denoise", ["SCUNet", "NAFNet", "DPIR"]);

    var upscaleModel = function () {
        return dropdownValues["Model"];
    };
    var interpolateModel = function () {
        return dropdownValues["Interpolate"];
    };
    var depthModel = function () {
        return dropdownValues["Depth"];
    };
    var encoderMethod = function () {
        return dropdownValues["Encoder"];
    };
    var resizeMethod = function () {
        return dropdownValues["Resize"];
    };
    var dedupMethod = function () {
        return dropdownValues["Dedup"];
    };
    var denoiseMethod = function () {
        return dropdownValues["Denoise"];
    };

    var panelCustomSettings = settingsWindow.add("panel", undefined, undefined, {
        name: "panelCustomSettings"
    });

    panelCustomSettings.text = "Custom FFMPEG Encoding Parameters";
    panelCustomSettings.orientation = "column";
    panelCustomSettings.alignChildren = ["left", "top"];

    var textCustomEncoder = panelCustomSettings.add("edittext", [0, 0, 202, 25], undefined, {
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

    buttonSettingsClose.onClick = function () {
        settingsWindow.hide();
    }

    checkIfPathSaved = function () {
        if (theAnimeScripterPath == "undefined" || theAnimeScripterPath == null) {
            alert("The Anime Scripter directory has not been selected, please go to settings");
            return;
        }
    }

    buttonOfflineMode.onClick = function () {
        checkIfPathSaved();
        var exeFile = theAnimeScripterPath + "\\main.exe";
        var exeFilePath = new File(exeFile);
        if (!exeFilePath.exists) {
            alert("Cannot find main.exe, please make sure you have selected the correct folder in settings!");
            return;
        }
        var attempt = [
            "cd", "\"" + theAnimeScripterPath + "\"",
            "&&",
            "\"" + exeFile + "\"",
            "--offline",
        ];
        var command = attempt.join(" ");
        callCommand(command);
    }


    buttonCheckForUpdate.onClick = function () {
        checkIfPathSaved();

        var exeFile = theAnimeScripterPath + "\\main.exe";
        var exeFilePath = new File(exeFile);
        if (!exeFilePath.exists) {
            alert("Cannot find main.exe, please make sure you have selected the correct folder in settings!");
            return;
        };

        try {
            var command = [
                "cd", "\"" + theAnimeScripterPath + "\"",
                "&&",
                "\"" + exeFile + "\"",
                "--update",
            ].join(" ");
            callCommand(command);
        } catch (error) {
            alert(error.toString());
        }

        var discordServer = "https://discord.gg/CdRD9GwS8J";

        var dialog2 = new Window('dialog', 'Open URL');
        dialog2.add('statictext', undefined, 'Do you want to join the Discord server?');

        dialog2.yesButton = dialog2.add('button', undefined, 'Yes', {
            name: 'yes'
        });
        dialog2.noButton = dialog2.add('button', undefined, 'No', {
            name: 'no'
        });

        dialog2.yesButton.onClick = function () {
            system.callSystem('cmd.exe /c start "" "' + discordServer + '"');
            dialog2.close();
        }

        dialog2.noButton.onClick = function () {
            dialog2.close();
        }

        dialog2.show();
    };

    buttonOutput.onClick = function () {
        var folder = Folder.selectDialog("Select Output folder");
        if (folder != null) {
            outputFolder = folder.fsName;
            app.settings.saveSetting(scriptName, "outputFolder", outputFolder);

            textOutputFolderValue.text = outputFolder;
        }
        alert("Output folder set to: " + outputFolder);
    };

    buttonFolder.onClick = function () {
        var folder = Folder.selectDialog("Select The Anime Scripter folder");
        if (folder != null) {
            theAnimeScripterPath = folder.fsName;
            app.settings.saveSetting(scriptName, "theAnimeScripterPath", theAnimeScripterPath);

            textTheAnimeScripterFolderValue.text = theAnimeScripterPath;
        }
        alert("The Anime Scripter folder set to: " + theAnimeScripterPath);
    };

    buttonSettings.onClick = function () {
        settingsWindow.show();
    };

    buttonAutoClip.onClick = function () {
        autoClipValue = 1;
        startChain();
    }

    buttonDepthMap.onClick = function () {
        depthValue = 1;
        startChain();
    }

    buttonSegment.onClick = function () {
        segmentValue = 1;
        startChain();
    }

    buttonStartProcess.onClick = function () {
        startChain();
    }

    buttonGetVideo.onClick = function () {
        if (textGetVideo.text == "Add Youtube URL") {
            alert("Please add a Youtube URL");
            return;
        }

        startDownload();
    }

    buttonPreRender.onClick = function () {
        preRender();
    }

    buttonCustomModel.onClick = function () {
        var customModel = File.openDialog("Select a Custom Model", "*.pth", false);
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
                system.callSystem(cmdCommand);
            } else {
                throw new Error("Command is undefined");
            }
        } catch (error) {
            return alert(error.toString());
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

        var minStartTime = Infinity;
        var maxEndTime = 0;
        for (var i = 0; i < selectedLayers.length; i++) {
            var layer = selectedLayers[i];
            var startTime = layer.startTime;
            var endTime = layer.outPoint;
            if (startTime < minStartTime) {
                minStartTime = startTime;
            }
            if (endTime > maxEndTime) {
                maxEndTime = endTime;
            }
        }

        var newCompDuration = maxEndTime - minStartTime;
        var newComp = app.project.items.addComp('New Composition', comp.width, comp.height, comp.pixelAspect, newCompDuration, comp.frameRate);

        for (var i = 0; i < selectedLayers.length; i++) {
            var layer = selectedLayers[i];
            var newLayer = newComp.layers.add(layer.source);
            newLayer.startTime = layer.startTime - minStartTime;
            newLayer.inPoint = layer.inPoint - minStartTime;
            newLayer.outPoint = layer.outPoint - minStartTime;
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
        outputModule.bitrate = '50';

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

        checkIfPathSaved();

        var exeFile = theAnimeScripterPath + "\\main.exe";
        var exeFilePath = new File(exeFile);
        if (!exeFilePath.exists) {
            alert("Cannot find main.exe, please make sure you have selected the correct folder in settings!");
            return;
        }

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
            outputName = outputFolder + "\\" + activeLayerName.replace(/\.[^\.]+$/, '') + "-" + randomNumbers + outputFileExtension;

            try {
                var attempt = [
                    "cd", "\"" + theAnimeScripterPath + "\"",
                    "&&",
                    "\"" + exeFile + "\"",
                    "--input", "\"" + activeLayerPath + "\"",
                    "--output", "\"" + outputName + "\"",
                    "--interpolate_factor", interpolateValue(),
                    "--interpolate_method", interpolateModel().toLowerCase(),
                    "--upscale_factor", upscaleValue(),
                    "--upscale_method", upscaleModel().toLowerCase(),
                    "--dedup_sens", dedupSensValue(),
                    "--dedup_method", dedupMethod().toLowerCase(),
                    "--half",
                    "--inpoint", sourceInPoint,
                    "--outpoint", sourceOutPoint,
                    "--sharpen_sens", sharpenSensValue(),
                    "--depth_method", depthModel().toLowerCase(),
                    "--encode_method", encoderMethod().toLowerCase(),
                    "--autoclip_sens", sceneChangeSensValue(),
                    "--resize_method", resizeMethod().toLowerCase(),
                    "--resize_factor", resizeValue(),
                    "--nt", threadsValue(),
                    "--denoise_method", denoiseMethod().toLowerCase(),
                    "--ae",
                ];

                if (checkboxInterpolateValue()) {
                    attempt.push("--interpolate");
                }

                if (checkboxUpscaleValue()) {
                    attempt.push("--upscale");
                }

                if (checkboxDeduplicateValue()) {
                    attempt.push("--dedup");
                }

                if (checkboxSharpenValue()) {
                    attempt.push("--sharpen");
                }

                if (checkboxEnsembleValue()) {
                    attempt.push("--ensemble");
                }

                if (checkboxResizeValue()) {
                    attempt.push("--resize");
                }

                if (checkboxDenoiseValue()) {
                    attempt.push("--denoise");
                }


                if (segmentValue == 1) {
                    attempt.push("--segment");
                }

                if (depthValue == 1) {
                    attempt.push("--depth");
                }

                if (autoClipValue == 1) {
                    attempt.push("--autoclip");
                }

                if (customModelPath && customModelPath !== "undefined") {
                    attempt.push("--custom_model", customModelPath);
                }

                if (textCustomEncoder.text && textCustomEncoder.text !== "") {
                    attempt.push("--custom_encoder", '"' + textCustomEncoder.text + '"');
                }

                var command = attempt.join(" ");
                callCommand(command);
            } catch (error) {
                alert(error);
            }

            while (true) {
                if (autoClipValue == 1) {
                    var autoClipLogPath = theAnimeScripterPath + "\\autoclipresults.txt";
                    var autoClipLog = new File(autoClipLogPath);
                    autoClipLog.open("r");

                    inPoint = layer.inPoint;
                    outPoint = layer.outPoint;

                    while (!autoClipLog.eof) {
                        var line = autoClipLog.readln();
                        var timestamp = parseFloat(line) + inPoint;
                        var duplicateLayer = layer.duplicate();
                        layer.outPoint = timestamp;
                        duplicateLayer.inPoint = timestamp;

                        layer = duplicateLayer;
                    }
                    autoClipLog.close();
                    break;
                } else {
                    var maxAttempts = 3;
                    var importSuccessful = false;
                    for (var attempt = 0; attempt < maxAttempts; attempt++) {
                        if (importSuccessful) {
                            break;
                        }
                        $.sleep(1000); // Sleeping for a second, metadata is not always written instantly
                        try {
                            var importOptions = new ImportOptions(File(outputName));
                            var importedFile = app.project.importFile(importOptions);
                            var inputLayer = comp.layers.add(importedFile);
                            inputLayer.startTime = layer.inPoint;
                            inputLayer.moveBefore(layer);
                            if (checkboxUpscaleValue() == true || checkboxResizeValue() == true) {
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
                            importSuccessful = true;
                        } catch (error) {
                            if (attempt == maxAttempts - 1) {
                                alert("Failed to import file after " + maxAttempts + " attempts: " + outputName);
                                alert("Error: " + error.toString());
                            }
                        }
                    }
                    break;
                }
            }
            i++;
        }
        autoClipValue = 0;
        segmentValue = 0;
        depthValue = 0;
    }

    function startDownload() {
        if (outputFolder == "undefined" || outputFolder == null) {
            alert("The output folder has not been selected, please go to settings");
            return;
        }

        checkIfPathSaved();

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
                "--input", "\"" + textGetVideo.text + "\"",
                "--ae",
            ];
            var command = attempt.join(" ");
            callCommand(command);
        } catch (error) {
            alert(error);
        }

        try {
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