var panelGlobal = this;
var TheAnimeScripter = (function() {

    /*
    Code for Import https://scriptui.joonas.me — (Triple click to select): 
    {"activeId":36,"items":{"item-0":{"id":0,"type":"Dialog","parentId":false,"style":{"enabled":true,"varName":"TheAnimeScripter","windowType":"Palette","creationProps":{"su1PanelCoordinates":false,"maximizeButton":false,"minimizeButton":false,"independent":false,"closeButton":true,"borderless":false,"resizeable":false},"text":"TheAnimeScripter","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["center","top"]}},"item-2":{"id":2,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":"panelChain","creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"Chain","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-4":{"id":4,"type":"Checkbox","parentId":30,"style":{"enabled":true,"varName":"checkboxUpscale","text":"","preferredSize":[0,0],"alignment":"center","helpTip":null,"checked":false}},"item-5":{"id":5,"type":"Checkbox","parentId":32,"style":{"enabled":true,"varName":"checkboxInterpolate","text":"","preferredSize":[0,0],"alignment":"center","helpTip":null}},"item-6":{"id":6,"type":"Checkbox","parentId":29,"style":{"enabled":true,"varName":"checkboxDeduplicate","text":"","preferredSize":[0,0],"alignment":"center","helpTip":null,"checked":false}},"item-15":{"id":15,"type":"Button","parentId":2,"style":{"enabled":true,"varName":"buttonStartProcess","text":"Start  Process","justify":"center","preferredSize":[104,0],"alignment":"center","helpTip":null}},"item-25":{"id":25,"type":"Button","parentId":26,"style":{"enabled":true,"varName":"buttonSettings","text":"Settings","justify":"center","preferredSize":[105,0],"alignment":"center","helpTip":null}},"item-26":{"id":26,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":"panelMore","creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"More","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-27":{"id":27,"type":"Button","parentId":26,"style":{"enabled":true,"varName":"buttonInfo","text":"Info","justify":"center","preferredSize":[105,0],"alignment":"center","helpTip":null}},"item-29":{"id":29,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":10,"alignChildren":["left","center"],"alignment":null}},"item-30":{"id":30,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":10,"alignChildren":["left","center"],"alignment":null}},"item-32":{"id":32,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":10,"alignChildren":["left","center"],"alignment":null}},"item-36":{"id":36,"type":"StaticText","parentId":32,"style":{"enabled":true,"varName":"textInterpolate","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Interpolate","justify":"center","preferredSize":[0,0],"alignment":"center","helpTip":null}},"item-37":{"id":37,"type":"StaticText","parentId":30,"style":{"enabled":true,"varName":"textUpscale","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Upscale","justify":"center","preferredSize":[0,0],"alignment":"center","helpTip":null}},"item-38":{"id":38,"type":"StaticText","parentId":29,"style":{"enabled":true,"varName":"textDeduplicate","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Deduplicate","justify":"left","preferredSize":[0,0],"alignment":null,"helpTip":null}},"item-39":{"id":39,"type":"Button","parentId":40,"style":{"enabled":false,"varName":"buttonDepthMap","text":"Depth Map","justify":"center","preferredSize":[105,0],"alignment":"center","helpTip":null}},"item-40":{"id":40,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":"panelExtra","creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"Extra","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-41":{"id":41,"type":"Button","parentId":40,"style":{"enabled":false,"varName":"buttonSegment","text":"Segment","justify":"center","preferredSize":[105,0],"alignment":null,"helpTip":null}}},"order":[0,2,15,29,6,38,30,4,37,32,5,36,40,39,41,26,27,25],"settings":{"importJSON":true,"indentSize":false,"cepExport":false,"includeCSSJS":true,"showDialog":true,"functionWrapper":true,"afterEffectsDockable":true,"itemReferenceList":"none"}}
    */

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
    var mainPyPath = "";
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

    // FUNCTIONS
    buttonSettings.onClick = function() {
        var settingsWindow = (function() {

            /*
            Code for Import https://scriptui.joonas.me — (Triple click to select): 
            {"activeId":17,"items":{"item-0":{"id":0,"type":"Dialog","parentId":false,"style":{"enabled":true,"varName":"settingsWindow","windowType":"Dialog","creationProps":{"su1PanelCoordinates":false,"maximizeButton":false,"minimizeButton":false,"independent":false,"closeButton":true,"borderless":false,"resizeable":true},"text":"Settings","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["center","top"]}},"item-1":{"id":1,"type":"StaticText","parentId":3,"style":{"enabled":true,"varName":"textInterpolationMultiplier","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Interpolation Multiplier","justify":"left","preferredSize":[172,0],"alignment":"center","helpTip":null}},"item-2":{"id":2,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":"generalPanel","creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"General","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-3":{"id":3,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-4":{"id":4,"type":"EditText","parentId":3,"style":{"enabled":true,"varName":"interpolationInt","creationProps":{"noecho":false,"readonly":false,"multiline":false,"scrollable":false,"borderless":false,"enterKeySignalsOnChange":false},"softWrap":false,"text":"2","justify":"center","preferredSize":[40,0],"alignment":"center","helpTip":null}},"item-5":{"id":5,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-6":{"id":6,"type":"StaticText","parentId":5,"style":{"enabled":true,"varName":"textUpscaleMultiplier","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Upscale Multiplier","justify":"left","preferredSize":[172,0],"alignment":null,"helpTip":null}},"item-7":{"id":7,"type":"EditText","parentId":5,"style":{"enabled":true,"varName":"upscaleInt","creationProps":{"noecho":false,"readonly":false,"multiline":false,"scrollable":false,"borderless":false,"enterKeySignalsOnChange":false},"softWrap":false,"text":"2","justify":"center","preferredSize":[40,0],"alignment":"top","helpTip":null}},"item-9":{"id":9,"type":"StaticText","parentId":2,"style":{"enabled":false,"varName":"textDedupSensitivity","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Deduplication Sensitivity","justify":"center","preferredSize":[0,0],"alignment":"center","helpTip":null}},"item-10":{"id":10,"type":"Slider","parentId":2,"style":{"enabled":false,"varName":"sliderDedupSens","preferredSize":[212,0],"alignment":"center","helpTip":""}},"item-11":{"id":11,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":null,"creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"Model Picker","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-12":{"id":12,"type":"Group","parentId":11,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-13":{"id":13,"type":"Group","parentId":11,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-14":{"id":14,"type":"StaticText","parentId":12,"style":{"enabled":true,"varName":"textUpscaleModel","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Upscale Model","justify":"left","preferredSize":[103,0],"alignment":null,"helpTip":null}},"item-15":{"id":15,"type":"DropDownList","parentId":12,"style":{"enabled":true,"varName":"dropdownModel","text":"DropDownList","listItems":"ShuffleCugan, -, UltraCompact, -, Compact, -, Cugan, -,  SwinIR","preferredSize":[109,0],"alignment":null,"selection":0,"helpTip":"Choose which model you want to utilize, ordered by speed, read more in INFO"}},"item-16":{"id":16,"type":"StaticText","parentId":13,"style":{"enabled":true,"varName":"cuganDenoiseText","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Cugan Denoise","justify":"left","preferredSize":[103,0],"alignment":null,"helpTip":null}},"item-17":{"id":17,"type":"DropDownList","parentId":13,"style":{"enabled":true,"varName":"dropdownCugan","text":"DropDownList","listItems":"No-Denoise, -, Conservative, -, Denoise1x, -, Denoise2x","preferredSize":[109,0],"alignment":null,"selection":0,"helpTip":null}},"item-18":{"id":18,"type":"Group","parentId":2,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-19":{"id":19,"type":"StaticText","parentId":18,"style":{"enabled":true,"varName":"textNumberOfThreads","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Number of Threads","justify":"left","preferredSize":[172,0],"alignment":null,"helpTip":null}},"item-20":{"id":20,"type":"EditText","parentId":18,"style":{"enabled":true,"varName":"numberOfThreadsInt","creationProps":{"noecho":false,"readonly":false,"multiline":false,"scrollable":false,"borderless":false,"enterKeySignalsOnChange":false},"softWrap":false,"text":"2","justify":"center","preferredSize":[40,0],"alignment":null,"helpTip":null}},"item-21":{"id":21,"type":"Group","parentId":11,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-22":{"id":22,"type":"StaticText","parentId":21,"style":{"enabled":true,"varName":"textSwinIr","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"SwinIR Model","justify":"left","preferredSize":[103,0],"alignment":null,"helpTip":null}},"item-23":{"id":23,"type":"DropDownList","parentId":21,"style":{"enabled":true,"varName":"dropdownSwinIr","text":"DropDownList","listItems":"Small, -, Medium, -, Large","preferredSize":[109,0],"alignment":null,"selection":0,"helpTip":null}},"item-25":{"id":25,"type":"Group","parentId":11,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":0,"alignChildren":["left","center"],"alignment":null}},"item-26":{"id":26,"type":"StaticText","parentId":25,"style":{"enabled":false,"varName":"textSegment","creationProps":{"truncate":"none","multiline":false,"scrolling":false},"softWrap":false,"text":"Segment Model","justify":"left","preferredSize":[103,0],"alignment":null,"helpTip":null}},"item-27":{"id":27,"type":"DropDownList","parentId":25,"style":{"enabled":false,"varName":"dropdwonSegment","text":"DropDownList","listItems":"isnet-anime,","preferredSize":[109,0],"alignment":null,"selection":0,"helpTip":null}},"item-28":{"id":28,"type":"Panel","parentId":0,"style":{"enabled":true,"varName":"panelOnFirstRun","creationProps":{"borderStyle":"etched","su1PanelCoordinates":false},"text":"On First Run","preferredSize":[0,0],"margins":10,"orientation":"column","spacing":10,"alignChildren":["left","top"],"alignment":null}},"item-29":{"id":29,"type":"Button","parentId":31,"style":{"enabled":true,"varName":"buttonOutput","text":"Set Output","justify":"center","preferredSize":[100,0],"alignment":null,"helpTip":null}},"item-30":{"id":30,"type":"Button","parentId":31,"style":{"enabled":true,"varName":"buttonMainPy","text":"Set Main.py","justify":"center","preferredSize":[101,0],"alignment":null,"helpTip":null}},"item-31":{"id":31,"type":"Group","parentId":28,"style":{"enabled":true,"varName":null,"preferredSize":[0,0],"margins":0,"orientation":"row","spacing":10,"alignChildren":["left","center"],"alignment":null}}},"order":[0,28,31,29,30,2,9,10,3,1,4,5,6,7,18,19,20,11,12,14,15,13,16,17,21,22,23,25,26,27],"settings":{"importJSON":true,"indentSize":false,"cepExport":false,"includeCSSJS":true,"showDialog":true,"functionWrapper":true,"afterEffectsDockable":true,"itemReferenceList":"none"}}
            */

            // SETTINGSWINDOW
            // ==============
            var settingsWindow = (panelGlobal instanceof Panel) ? panelGlobal : new Window("palette", undefined, undefined, {
                resizeable: true
            });
            if (!(panelGlobal instanceof Panel)) settingsWindow.text = "Settings";
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

            var buttonOutput = group1.add("button", undefined, undefined, {
                name: "buttonOutput"
            });
            buttonOutput.text = "Set Output";
            buttonOutput.preferredSize.width = 100;

            var buttonMainPy = group1.add("button", undefined, undefined, {
                name: "buttonMainPy"
            });
            buttonMainPy.text = "Set Main.py";
            buttonMainPy.preferredSize.width = 101;

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

            var interpolationInt = group2.add('edittext {justify: "center", properties: {name: "interpolationInt"}}');
            interpolationInt.text = "2";
            interpolationInt.preferredSize.width = 40;
            interpolationInt.alignment = ["left", "center"];

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

            var upscaleInt = group3.add('edittext {justify: "center", properties: {name: "upscaleInt"}}');
            upscaleInt.text = "2";
            upscaleInt.preferredSize.width = 40;
            upscaleInt.alignment = ["left", "top"];

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
            textNumberOfThreads.text = "Number of Threads";
            textNumberOfThreads.preferredSize.width = 172;

            var numberOfThreadsInt = group4.add('edittext {justify: "center", properties: {name: "numberOfThreadsInt"}}');
            numberOfThreadsInt.text = "2";
            numberOfThreadsInt.preferredSize.width = 40;

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

            var dropdownModel_array = ["ShuffleCugan", "-", "UltraCompact", "-", "Compact", "-", "Cugan", "-", "SwinIR"];
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
            textSwinIr.text = "SwinIR Model";
            textSwinIr.preferredSize.width = 103;

            var dropdownSwinIr_array = ["Small", "-", "Medium", "-", "Large"];
            var dropdownSwinIr = group7.add("dropdownlist", undefined, undefined, {
                name: "dropdownSwinIr",
                items: dropdownSwinIr_array
            });
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

            settingsWindow.layout.layout(true);
            settingsWindow.layout.resize();
            settingsWindow.onResizing = settingsWindow.onResize = function() {
                this.layout.resize();
            }

            buttonOutput.onClick = function() {
                try {
                    var folder = new Folder()
                    var outputFolder = folder.selectDlg("Select an output directory");
                    if (outputFolder != null) {
                        app.settings.saveSetting("AnimeScripter", "outputFolder", outputFolder.fsName);
                    }
                    alert("successfully saved path");
                } catch (error) {
                    alert(error);
                }
            }

            buttonMainPy.onClick = function() {
                try {
                    var mainPyPath = File.openDialog("Select the main.py file");
                    if (mainPyPath != null) {
                        app.settings.saveSetting("AnimeScripter", "mainPyPath", mainPyPath.fsName);
                    }
                    alert("successfully saved path");
                } catch (error) {
                    alert(error);
                }
            }

            interpolationInt.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "interpolationInt", interpolationInt.text);
            }

            upscaleInt.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "upscaleInt", upscaleInt.text);
            }

            numberOfThreadsInt.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "numberOfThreadsInt", numberOfThreadsInt.text);
            }

            dropdownModel.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "dropdownModel", dropdownModel.selection.index);
            }

            dropdownCugan.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "dropdownCugan", dropdownCugan.selection.index);
            }

            dropdownSwinIr.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "dropdownSwinIr", dropdownSwinIr.selection.index);
            }

            dropdwonSegment.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "dropdwonSegment", dropdwonSegment.selection.index);
            }

            sliderDedupSens.onChange = function() {
                app.settings.saveSetting("AnimeScripter", "sliderDedupSens", sliderDedupSens.value);
            }

            if (settingsWindow instanceof Window) settingsWindow.show();
            return settingsWindow;

        }());
    }

    buttonStartProcess.onClick = function() {
        if (checkboxDeduplicate.value == false && checkboxUpscale.value == false && checkboxInterpolate.value == false) {
            alert("Please select at least one process");
            return;
        }
        chain_models();
    }

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

    function check_weights(scriptPath) {
        // Checking if the user has downloaded the Cugan Models
        weightsPath = scriptPath + "\\src\\cugan\\weights\\";

        // TO:DO, add checking for each model path
        var weightsFile = new File(weightsPath);
        if (!weightsFile.exists) {
            alert("Models folder(s) not found, please make sure you have downloaded the models, run setup.bat or python download_models.py in the script folder and try again");
            return;
        }
    }

    function handleTrimmedInput(inPoint, outPoint, layer, activeLayerPath, activeLayerName, outputFolder, scriptPath, module) {
        var startTime = layer.startTime;
        var newInPoint = inPoint - startTime;
        var newOutPoint = outPoint - startTime;

        output_name = outputFolder + "\\" + activeLayerName + "_temp.mp4";
        var trimInputPath = scriptPath + "\\src\\trim_input.py"

        command = "cd \"" + scriptPath + "\" && python \"" + trimInputPath + "\" -ss " + newInPoint + " -to " + newOutPoint + " -i \"" + activeLayerPath + "\" -o \"" + output_name + "\"";
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

    function chain_models() {
        try {
            var outputFolder = app.settings.haveSetting("AnimeScripter", "outputFolder") ? app.settings.getSetting("AnimeScripter", "outputFolder") : "";
            var mainPyPath = app.settings.haveSetting("AnimeScripter", "mainPyPath") ? app.settings.getSetting("AnimeScripter", "mainPyPath") : "";
            var interpolationInt = app.settings.haveSetting("AnimeScripter", "interpolationInt") ? app.settings.getSetting("AnimeScripter", "interpolationInt") : defaultInterpolateInt;
            var numberOfThreadsInt = app.settings.haveSetting("AnimeScripter", "numberOfThreadsInt") ? app.settings.getSetting("AnimeScripter", "numberOfThreadsInt") : defaultNrThreads;
            var upscaleInt = app.settings.haveSetting("AnimeScripter", "upscaleInt") ? app.settings.getSetting("AnimeScripter", "upscaleInt") : defaultUpscaleInt;
            var dropdownCugan = app.settings.haveSetting("AnimeScripter", "dropdownCugan") ? app.settings.getSetting("AnimeScripter", "dropdownCugan") : defaultCugan;
            var dropdownModel = app.settings.haveSetting("AnimeScripter", "dropdownModel") ? app.settings.getSetting("AnimeScripter", "dropdownModel") : defaultUpscaler;
            var dropdownSwinIr = app.settings.haveSetting("AnimeScripter", "dropdownSwinIr") ? app.settings.getSetting("AnimeScripter", "dropdownSwinIr") : defaultSwinIR;
            var dropdwonSegment = app.settings.haveSetting("AnimeScripter", "dropdwonSegment") ? app.settings.getSetting("AnimeScripter", "dropdwonSegment") : defaultSegment;

            dropdownModel = dropdownModel_array[dropdownModel];
            dropdwonCugan = dropdownCugan_array[dropdownCugan];
            dropdownSwinIr = dropdownSwinIr_array[dropdownSwinIr];
            dropdwonSegment = dropdwonSegment_array[dropdwonSegment];

        } catch (error) {
            alert("Something went wrong trying to get the data from settings, chain_models(), please contact me on discord");
            return;
        }

        alert("I AM HERE")
        if (((!app.project) || (!app.project.activeItem)) || (app.project.activeItem.selectedLayers.length < 1)) {
            alert("Please select one layer.");
            return;
        }

        if (outputFolder == "" || outputFolder == null) {
            alert("No output directory selected");
            return;
        }

        if (mainPyPath == "" || mainPyPath == null) {
            alert("No main.py file selected");
            return;
        }

        var pyfile = File(mainPyPath);
        var scriptPath = pyfile.parent.fsName;

        // Checking if the user has downloaded the models
        check_weights(scriptPath);

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
                var result = handleTrimmedInput(inPoint, outPoint, layer, activeLayerPath, activeLayerName, outputFolder, scriptPath, module)
                activeLayerPath = result[0];
                output_name = result[1];
                removeFile = result[2];
                temp_output_name = output_name;
            } else {
                temp_output_name = outputFolder + "\\" + activeLayerName
                output_name = outputFolder + "\\" + activeLayerName
            }

            if (dedupCheckmark.value == true) {
                output_name = output_name + "_de" + ".m4v";
                command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type dedup -kind_model " + "ffmpeg" + " -output \"" + output_name + "\"";

                callCommand(command);
                // For removing the residue
                if (upcsaleCheckmark.value == true || interpolateCheckmark.value == true) {
                    var remFile_2 = new File(output_name);
                }
            }

            if (upcsaleCheckmark.value == true) {
                if (output_name !== temp_output_name) {
                    activeLayerPath = output_name;
                    output_name = output_name.replace(".m4v", '')
                    output_name = output_name + "_up" + ".m4v";
                } else {
                    output_name = output_name + "_up" + ".m4v";
                }
                if (DropdownUpscaler == "ShuffleCugan") {
                    command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type shufflecugan -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt + " -output \"" + output_name + "\"";
                } else if (DropdownUpscaler == "Cugan") {
                    command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type cugan -nt " + NumberOfThreadsInt + " -kind_model " + DropdownCugan + " -multi " + UpscaleInt + " -output \"" + output_name + "\"";
                } else if (DropdownUpscaler == "UltraCompact") {
                    command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type ultracompact -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt + " -output \"" + output_name + "\"";
                } else if (DropdownUpscaler == "Compact") {
                    command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type compact -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt + " -output \"" + output_name + "\"";
                } else if (DropdownUpscaler == "Swwinir") {
                    command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type swinir -nt " + NumberOfThreadsInt + " -kind_model " + DropdownSwinIr + " -multi " + UpscaleInt + " -output \"" + output_name + "\"";
                } else {
                    alert("No model has been selected, weird, please try setting a model again, if it doesn't work contact me on the discord server")
                    return;
                }
                callCommand(command);

                // For removing the residue
                if (interpolateCheckmark.value == true) {
                    var remFile_3 = new File(output_name);
                }
            }

            if (interpolateCheckmark.value == true) {
                if (output_name !== temp_output_name) {
                    activeLayerPath = output_name;
                    output_name = output_name.replace(".m4v", '')
                    output_name = output_name + "_int" + ".m4v";
                } else {
                    output_name = output_name + "_int" + ".m4v";
                }
                command = "cd \"" + scriptPath + "\" && python \"" + mainPyPath + "\" -video \"" + activeLayerPath + "\" -model_type interpolation -multi " + InterpolationInt + " -output \"" + output_name + "\"";
                callCommand(command);
            }

            if (removeFile && removeFile.exists) {
                try {
                    removeFile.remove();
                } catch (error) {
                    alert(error);
                    alert("There might have been a problem removing one of the temp files. Do you have admin permissions?");
                }
            }

            if (remFile_2 && remFile_2.exists) {
                try {
                    remFile_2.remove();
                } catch (error) {
                    alert(error);
                    alert("There might have been a problem removing one of the temp files. Do you have admin permissions?");
                }
            }

            if (remFile_3 && remFile_3.exists) {
                try {
                    remFile_3.remove();
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

                    if (upcsaleCheckmark == true) {
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
    }

    if (TheAnimeScripter instanceof Window) TheAnimeScripter.show();
    return TheAnimeScripter;

}());