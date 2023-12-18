var panelGlobal = this;
var TheAnimeScripter = (function() {

    var scriptName = "AnimeScripter";
    var scriptVersion = "0.1.0";
    var scriptAuthor = "Nilas";
    var scriptURL = "https://github.com/NevermindNilas/TheAnimeScripter"
    var discordServer = "https://discord.gg/CdRD9GwS8J"

    //DEFAULT VALUES
    var defaultUpscaler = "ShuffleCugan";
    var defaultNrThreads = 2;
    var defaultCugan = "no-denoise" || 0;
    var defaultSwinIR = "small" || 4;
    var defaultSegment = "isnet-anime" || 0;
    var defaultInterpolateInt = 2;
    var defaultUpscaleInt = 2;
    var TheAnimeScripterPath = "";
    var outputFolder = "";

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

    var checkboxDeduplicate = group1.add("checkbox", undefined, undefined, {
        name: "checkboxDeduplicate"
    });
    checkboxDeduplicate.alignment = ["left", "center"];

    var textDeduplicate = group1.add("statictext", undefined, undefined, {
        name: "textDeduplicate"
    });
    textDeduplicate.text = "Deduplicate";

    // GROUP2
    // ======
    var group2 = panelChain.add("group", undefined, {
        name: "group2"
    });
    group2.orientation = "row";
    group2.alignChildren = ["left", "center"];
    group2.spacing = 10;
    group2.margins = 0;

    var checkboxUpscale = group2.add("checkbox", undefined, undefined, {
        name: "checkboxUpscale"
    });
    checkboxUpscale.alignment = ["left", "center"];

    var textUpscale = group2.add("statictext", undefined, undefined, {
        name: "textUpscale"
    });
    textUpscale.text = "Upscale";
    textUpscale.justify = "center";
    textUpscale.alignment = ["left", "center"];

    // GROUP3
    // ======
    var group3 = panelChain.add("group", undefined, {
        name: "group3"
    });
    group3.orientation = "row";
    group3.alignChildren = ["left", "center"];
    group3.spacing = 10;
    group3.margins = 0;

    var checkboxInterpolate = group3.add("checkbox", undefined, undefined, {
        name: "checkboxInterpolate"
    });
    checkboxInterpolate.alignment = ["left", "center"];

    var textInterpolate = group3.add("statictext", undefined, undefined, {
        name: "textInterpolate"
    });
    textInterpolate.text = "Interpolate";
    textInterpolate.justify = "center";
    textInterpolate.alignment = ["left", "center"];

    // PANELEXTRA
    // ==========
    var panelExtra = TheAnimeScripter.add("panel", undefined, undefined, {
        name: "panelExtra"
    });
    panelExtra.text = "Extra";
    panelExtra.orientation = "column";
    panelExtra.alignChildren = ["left", "top"];
    panelExtra.spacing = 10;
    panelExtra.margins = 10;

    var buttonDepthMap = panelExtra.add("button", undefined, undefined, {
        name: "buttonDepthMap"
    });
    buttonDepthMap.enabled = false;
    buttonDepthMap.text = "Depth Map";
    buttonDepthMap.preferredSize.width = 105;
    buttonDepthMap.alignment = ["center", "top"];

    var buttonSegment = panelExtra.add("button", undefined, undefined, {
        name: "buttonSegment"
    });
    buttonSegment.enabled = false;
    buttonSegment.text = "Segment";
    buttonSegment.preferredSize.width = 105;

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
    // INFO FUNCTION
    buttonInfo.onClick = function() {
        var dialog = (function() {

            /*
            Code for Import https://scriptui.joonas.me — (Triple click to select): 
            {"activeId":4,"items":{"item-0":{"id":0,"type":"Dialog","parentId":false,"style":{"enabled":true,"varName":null,"windowType":"Dialog","creationProps":{"su1PanelCoordinates":false,"maximizeButton":false,"minimizeButton":false,"independent":false,"closeButton":true,"borderless":false,"resizeable":false},"text":"Dialog","preferredSize":[0,0],"margins":16,"orientation":"column","spacing":10,"alignChildren":["center","top"]}},"item-1":{"id":1,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":null,"creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"Socials","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-2":{"id":2,"type":"StaticText","parentId":1,"style":{"enabled":true,"varName":null,"creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Github:","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}},"item-3":{"id":3,"type":"StaticText","parentId":1,"style":{"enabled":true,"varName":null,"creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Discord:","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}},"item-4":{"id":4,"type":"StaticText","parentId":1,"style":{"enabled":true,"varName":null,"creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"https://github.com/NevermindNilas/TheAnimeScripter","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}},"item-5":{"id":5,"type":"StaticText","parentId":1,"style":{"enabled":true,"varName":null,"creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"https://discord.gg/SCs9WpjWCD","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}},"item-6":{"id":6,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":null,"creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"Certain Limitations","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-7":{"id":7,"type":"StaticText","parentId":6,"style":{"enabled":true,"varName":null,"creationProps":{},"softWrap":false,"text":"For now, the script only works with PASCAL ( GTX 1000 series ) \nand above or any modern CPU. AMD users will fallback to CPU\nwhich may not be ideal in terms of speed, AMD compatible \nmodels will be coming soon\n\nOn the first run of the script, you must set the Output Folder\nand the location of the script.\n\nShuffleCUGAN Supports only 2x upscaling and a maximum of\n1920x1080 input, meaning the output will be at most 4k\n\nIn terms of speed, the fastest one should be ShuffleCugan,\nfollowed by Ultracompact, Compact and Cugan.\n\nPlease feel free to report any bugs, issues and to suggest \nenhancements on the discord server. ","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}}},"order":[0,1,2,4,3,5,6,7],"settings":{"importJSON":true,"indentSize":false,"cepExport":false,"includeCSSJS":true,"showDialog":true,"functionWrapper":true,"afterEffectsDockable":false,"itemReferenceList":"None"}}
            */

            // DIALOG
            // ======
            var dialog = new Window("dialog");
            dialog.text = "Dialog";
            dialog.orientation = "column";
            dialog.alignChildren = ["center", "top"];
            dialog.spacing = 10;
            dialog.margins = 16;

            // PANEL1
            // ======
            var panel1 = dialog.add("panel", undefined, undefined, {
                name: "panel1"
            });
            panel1.text = "Socials";
            panel1.orientation = "column";
            panel1.alignChildren = ["left", "top"];
            panel1.spacing = 10;
            panel1.margins = 10;

            var statictext1 = panel1.add("statictext", undefined, undefined, {
                name: "statictext1"
            });
            statictext1.text = "Github:";

            var statictext2 = panel1.add("statictext", undefined, undefined, {
                name: "statictext2"
            });
            statictext2.text = "https://github.com/NevermindNilas/TheAnimeScripter";

            var statictext3 = panel1.add("statictext", undefined, undefined, {
                name: "statictext3"
            });
            statictext3.text = "Discord:";

            var statictext4 = panel1.add("statictext", undefined, undefined, {
                name: "statictext4"
            });
            statictext4.text = "https://discord.gg/SCs9WpjWCD";

            // PANEL2
            // ======
            var panel2 = dialog.add("panel", undefined, undefined, {
                name: "panel2"
            });
            panel2.text = "Certain Limitations";
            panel2.orientation = "column";
            panel2.alignChildren = ["left", "top"];
            panel2.spacing = 10;
            panel2.margins = 10;

            var statictext5 = panel2.add("group", undefined, {
                name: "statictext5"
            });
            statictext5.getText = function() {
                var t = [];
                for (var n = 0; n < statictext5.children.length; n++) {
                    var text = statictext5.children[n].text || '';
                    if (text === '') text = ' ';
                    t.push(text);
                }
                return t.join('\n');
            };
            statictext5.orientation = "column";
            statictext5.alignChildren = ["left", "center"];
            statictext5.spacing = 0;

            statictext5.add("statictext", undefined, "For now, the script only works with PASCAL ( GTX 1000 series ) ");
            statictext5.add("statictext", undefined, "and above or any modern CPU. AMD users will fallback to CPU");
            statictext5.add("statictext", undefined, "which may not be ideal in terms of speed, AMD compatible ");
            statictext5.add("statictext", undefined, "models will be coming soon ");
            statictext5.add("statictext", undefined, "");
            statictext5.add("statictext", undefined, "On the first run of the script, you must set the Output Folder");
            statictext5.add("statictext", undefined, "and the location of the script. ");
            statictext5.add("statictext", undefined, "");
            statictext5.add("statictext", undefined, "ShuffleCUGAN Supports only 2x upscaling and a maximum of");
            statictext5.add("statictext", undefined, "1920x1080 input, meaning the output will be at most 4k ");
            statictext5.add("statictext", undefined, "");
            statictext5.add("statictext", undefined, "In terms of speed, the fastest one should be ShuffleCugan,");
            statictext5.add("statictext", undefined, "followed by Ultracompact, Compact and Cugan. ");
            statictext5.add("statictext", undefined, "");
            statictext5.add("statictext", undefined, "Please feel free to report any bugs, issues and to suggest ");
            statictext5.add("statictext", undefined, "enhancements on the discord server. ");

            dialog.show();

            return dialog;

        }());
    }

    // SETTINGS FUNCTIONS
    buttonSettings.onClick = function() {
        var settingsWindow = (function() {

            // SETTINGSWINDOW
            // ==============
            var settingsWindow = new Window("dialog", undefined, undefined, {
                resizeable: true
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
            buttonFolder.text = "Set Folder";
            buttonFolder.preferredSize.width = 100;

            var buttonOutput = group1.add("button", undefined, undefined, {
                name: "buttonOutput"
            });
            buttonOutput.text = "Set Output";
            buttonOutput.preferredSize.width = 101;

            // GENERALPANEL
            // ============
            var generalPanel = settingsWindow.add("panel", undefined, undefined, {
                name: "generalPanel"
            });
            generalPanel.text = "General";
            generalPanel.orientation = "column";
            generalPanel.alignChildren = ["left", "top"];
            generalPanel.spacing = 10;
            generalPanel.margins = 10;

            var textDedupSensitivity = generalPanel.add("statictext", undefined, undefined, {
                name: "textDedupSensitivity"
            });
            textDedupSensitivity.enabled = false;
            textDedupSensitivity.text = "Deduplication Sensitivity";
            textDedupSensitivity.justify = "center";
            textDedupSensitivity.alignment = ["center", "top"];

            var sliderDedupSens = generalPanel.add("slider", undefined, undefined, undefined, undefined, {
                name: "sliderDedupSens"
            });
            sliderDedupSens.enabled = false;
            sliderDedupSens.minvalue = 0;
            sliderDedupSens.maxvalue = 100;
            sliderDedupSens.value = 50;
            sliderDedupSens.preferredSize.width = 212;
            sliderDedupSens.alignment = ["center", "top"];

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

            var textNumberOfThreads = group4.add("statictext", undefined, undefined, {
                name: "textNumberOfThreads"
            });
            textNumberOfThreads.enabled = false;
            textNumberOfThreads.text = "Number of Threads";
            textNumberOfThreads.preferredSize.width = 172;

            var intNumberOfThreads = group4.add('edittext {justify: "center", properties: {name: "intNumberOfThreads"}}');
            intNumberOfThreads.enabled = false;
            intNumberOfThreads.text = "1";
            intNumberOfThreads.preferredSize.width = 40;

            // PANEL1
            // ======
            var panel1 = settingsWindow.add("panel", undefined, undefined, {
                name: "panel1"
            });
            panel1.text = "Model Picker";
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

            var dropdownModel_array = ["ShuffleCugan", "-", "UltraCompact", "-", "Compact", "-", "Cugan", ""];
            var dropdownModel = group5.add("dropdownlist", undefined, undefined, {
                name: "dropdownModel",
                items: dropdownModel_array
            });
            dropdownModel.helpTip = "Choose which model you want to utilize, ordered by speed, read more in INFO";
            dropdownModel.selection = 0;
            dropdownModel.preferredSize.width = 109;

            // GROUP6
            // ======
            var group6 = panel1.add("group", undefined, {
                name: "group6"
            });
            group6.orientation = "row";
            group6.alignChildren = ["left", "center"];
            group6.spacing = 0;
            group6.margins = 0;

            var cuganDenoiseText = group6.add("statictext", undefined, undefined, {
                name: "cuganDenoiseText"
            });
            cuganDenoiseText.text = "Cugan Denoise";
            cuganDenoiseText.preferredSize.width = 103;

            var dropdownCugan_array = ["No-Denoise", "-", "Conservative", "-", "Denoise1x", "-", "Denoise2x"];
            var dropdownCugan = group6.add("dropdownlist", undefined, undefined, {
                name: "dropdownCugan",
                items: dropdownCugan_array
            });
            dropdownCugan.selection = 0;
            dropdownCugan.preferredSize.width = 109;

            // GROUP7
            // ======
            var group7 = panel1.add("group", undefined, {
                name: "group7"
            });
            group7.orientation = "row";
            group7.alignChildren = ["left", "center"];
            group7.spacing = 0;
            group7.margins = 0;

            var textSwinIr = group7.add("statictext", undefined, undefined, {
                name: "textSwinIr"
            });
            textSwinIr.enabled = false;
            textSwinIr.text = "SwinIR Model";
            textSwinIr.preferredSize.width = 103;

            var dropdownSwinIr_array = ["Small", "-", "Medium", "-", "Large"];
            var dropdownSwinIr = group7.add("dropdownlist", undefined, undefined, {
                name: "dropdownSwinIr",
                items: dropdownSwinIr_array
            });
            dropdownSwinIr.enabled = false;
            dropdownSwinIr.selection = 0;
            dropdownSwinIr.preferredSize.width = 109;

            // GROUP8
            // ======
            var group8 = panel1.add("group", undefined, {
                name: "group8"
            });
            group8.orientation = "row";
            group8.alignChildren = ["left", "center"];
            group8.spacing = 0;
            group8.margins = 0;

            var textSegment = group8.add("statictext", undefined, undefined, {
                name: "textSegment"
            });
            textSegment.enabled = false;
            textSegment.text = "Segment Model";
            textSegment.preferredSize.width = 103;

            var dropdwonSegment_array = ["isnet-anime", ""];
            var dropdwonSegment = group8.add("dropdownlist", undefined, undefined, {
                name: "dropdwonSegment",
                items: dropdwonSegment_array
            });
            dropdwonSegment.enabled = false;
            dropdwonSegment.selection = 0;
            dropdwonSegment.preferredSize.width = 109;

            // GROUP9
            // ======
            var group9 = panel1.add("group", undefined, {
                name: "group9"
            });
            group9.orientation = "row";
            group9.alignChildren = ["left", "center"];
            group9.spacing = 0;
            group9.margins = 0;

            var textDedupMethod = group9.add("statictext", undefined, undefined, {
                name: "textDedupMethod"
            });
            textDedupMethod.enabled = false;
            textDedupMethod.text = "Dedup Method";
            textDedupMethod.preferredSize.width = 103;

            var dropdown1_array = ["FFmpeg", "-", "SSIM", "-", "MSE"];
            var dropdown1 = group9.add("dropdownlist", undefined, undefined, {
                name: "dropdown1",
                items: dropdown1_array
            });
            dropdown1.enabled = false;
            dropdown1.selection = 0;
            dropdown1.preferredSize.width = 109;


            //FUNCTIONS
            buttonFolder.onClick = function() {
                var folder = Folder.selectDialog("Select The Anime Scripter folder");
                if (folder != null) {
                    TheAnimeScripterPath = folder.fsName;
                }
            };

            buttonOutput.onClick = function() {
                var folder = Folder.selectDialog("Select Output folder");
                if (folder != null) {
                    outputFolder = folder.fsName;
                }
            };

            intInterpolate.onChange = function() {
                app.settings.saveSetting(scriptName, "intInterpolate", intInterpolate.text);
            }

            intUpscale.onChange = function() {
                app.settings.saveSetting(scriptName, "intUpscale", intUpscale.text);
            }

            intNumberOfThreads.onChange = function() {
                app.settings.saveSetting(scriptName, "intNumberOfThreads", intNumberOfThreads.text);
            }

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

            sliderDedupSens.onChange = function() {
                app.settings.saveSetting(scriptName, "sliderDedupSens", sliderDedupSens.value);
            }

            settingsWindow.show();
            return settingsWindow;

        }());
    };

    if (TheAnimeScripter instanceof Window) TheAnimeScripter.show();

    function callCommand(command) {
        try {
            var cmdCommand = 'cmd.exe /c "' + command;
            system.callSystem(cmdCommand);

            // Added because the metadata would only finish writing after the script was done, I assume.
            $.sleep(500);
        } catch (error) {
            alert(error);
            alert("Something went wrong trying to process the chain, please contact me on discord")
        }
    }

    function check_weights(TheAnimeScripterPath) {
        // Checking if the user has downloaded the Cugan Models
        weightsPath = TheAnimeScripterPath + "\\src\\cugan\\weights\\";

        // TO:DO, add checking for each model path
        var weightsFile = new File(weightsPath);
        if (!weightsFile.exists) {
            alert("Models folder(s) not found, please make sure you have downloaded the models, run setup.bat or python download_models.py in the script folder and try again");
            return;
        }
    }

    function handleTrimmedInput(inPoint, outPoint, layer, activeLayerPath, activeLayerName, outputFolder, TheAnimeScripterPath, module) {
        var startTime = layer.startTime;
        var newInPoint = inPoint - startTime;
        var newOutPoint = outPoint - startTime;

        output_name = outputFolder + "\\" + activeLayerName + "_temp.mp4";
        var trimInputPath = TheAnimeScripterPath + "\\src\\trim_input.py"

        command = "cd \"" + TheAnimeScripterPath + "\" && python \"" + trimInputPath + "\" -ss " + newInPoint + " -to " + newOutPoint + " -i \"" + activeLayerPath + "\" -o \"" + output_name + "\"";
        cmdCommand = 'cmd.exe /c "' + command;

        system.callSystem(cmdCommand)

        activeLayerPath = output_name;

        // This is for removing the temp file that was created
        var removeFile = new File(activeLayerPath);


        output_name = output_name.replace("_temp.mp4", '')

        if (module !== "chain") {
            var randomNumber = Math.floor(Math.random() * 10000);
            output_name = output_name + "_" + module + "_" + randomNumber + ".m4v";
        }

        return [activeLayerPath, output_name, removeFile]
    }

    function start_chain() {
        try {
            var outputFolder = app.settings.getSetting(scriptName, "outputFolder") ? app.settings.getSetting(scriptName, "outputFolder") : "";
            var TheAnimeScripterPath = app.settings.getSetting(scriptName, "TheAnimeScripterPath") ? app.settings.getSetting(scriptName, "TheAnimeScripterPath") : "";
            var intInterpolate = app.settings.getSetting(scriptName, "intInterpolate") ? app.settings.getSetting(scriptName, "intInterpolate") : defaultInterpolateInt;
            var intUpscale = app.settings.getSetting(scriptName, "intUpscale") ? app.settings.getSetting(scriptName, "intUpscale") : defaultUpscaleInt;
            var dropdownModel = app.settings.getSetting(scriptName, "dropdownModel") ? app.settings.getSetting(scriptName, "dropdownModel") : defaultUpscaler;
            var dropdownCugan = app.settings.getSetting(scriptName, "dropdownCugan") ? app.settings.getSetting(scriptName, "dropdownCugan") : defaultCugan;
            var dropdownSwinIr = app.settings.getSetting(scriptName, "dropdownSwinIr") ? app.settings.getSetting(scriptName, "dropdownSwinIr") : defaultSwinIR;
            var dropdwonSegment = app.settings.getSetting(scriptName, "dropdwonSegment") ? app.settings.getSetting(scriptName, "dropdwonSegment") : defaultSegment;
            var intInterpolate = app.settings.getSetting(scriptName, "intInterpolate") ? app.settings.getSetting(scriptName, "intInterpolate") : defaultInterpolateInt;

            dropdownModel = dropdownModel_array[dropdownModel];
            dropdownCugan = dropdownCugan_array[dropdownCugan];
            dropdownSwinIr = dropdownSwinIr_array[dropdownSwinIr];
            dropdwonSegment = dropdwonSegment_array[dropdwonSegment];

        } catch (error) {
            alert("Something went wrong, please make sure you have set the Output Folder and the location of the script");
            return;
        }

        if (((!app.project) || (!app.project.activeItem)) || (app.project.activeItem.selectedLayers.length < 1)) {
            alert("Please select one layer.");
            return;
        }

        if (outputFolder == "" || outputFolder == null) {
            alert("The Output folder has not been selected, please go to settings");
            return;
        }

        if (TheAnimeScripterPath == "" || TheAnimeScripterPath == null) {
            alert("The Anime Scripter directory has not been selected, please go to settings");
            return;
        }
        
        var pyFile = TheAnimeScripterPath + "chain.py";
        var activeItem = app.project.activeItem;

        check_weights(TheAnimeScripterPath);

        var comp = app.project.activeItem;
        for (var i = 0; i < comp.selectedLayers.length; i++) {
            var layer = comp.selectedLayers[i];
            var activeLayerPath = layer.source.file.fsName;
            var activeLayerName = layer.name;

            var inPoint = layer.inPoint;
            var outPoint = layer.outPoint;
            var duration = outPoint - inPoint;

            if (duration !== layer.source.duration) {
                module = "chain";
                var result = handleTrimmedInput(inPoint, outPoint, layer, activeLayerPath, activeLayerName, outputFolder, TheAnimeScripterPath, module);
                activeLayerPath = result[0];
                output_name = result[1];
                removeFile = result[2];
                temp_output_name = output_name;
            } else {
                temp_output_name = outputFolder + "\\" + activeLayerName
                output_name = outputFolder + "\\" + activeLayerName
            }
            command = `cd "${TheAnimeScripterPath}" && python "${pyFile}" -i "${activeLayerPath}" -o "${output_name}" -int ${interpolateCheckmark} -intfactor ${intInterpolate} -ups ${upscaleCheckmark} -upsfactor ${intUpscale} -upsmethod "${dropdownModel}" -cugan "${dropdownCugan}" --nt ${intNumberOfThreads} -de ${deduplicateCheckmark} -dedupsens ${sliderDedupSens.value}`;
            callCommand(command);

            if (removeFile && removeFile.exists) {
                try {
                    removeFile.remove();
                } catch (error) {
                    alert(error);
                    alert("There might have been a problem removing one of the temp files. Do you have admin permissions?");
                }
            }

            importFile();
            function importFile() {
                try {
                    var importOptions = new ImportOptions(File(output_name));
                    var importedFile = app.project.importFile(importOptions);
                    var inputLayer = comp.layers.add(importedFile);
                    inputLayer.moveBefore(layer);

                    if (upscaleCheckmark == true) {
                        var compWidth = comp.width;
                        var compHeight = comp.height;
                        var layerWidth = inputLayer.source.width;
                        var layerHeight = inputLayer.source.height;
                        var scaleX = (compWidth / layerWidth) * 100;
                        var scaleY = (compHeight / layerHeight) * 100;
                        inputLayer.property("Scale").setValue([scaleX, scaleY, 100]);
                    }

                } catch (error) {
                    alert(error);
                    alert("Something went wrong trying to import the file, please contact me on discord")
                }
            }
        }
    return TheAnimeScripter;

}());