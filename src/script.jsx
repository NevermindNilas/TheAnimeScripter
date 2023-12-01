var panelGlobal = this;
var dialog = (function () {
	var scriptName = "AnimeScripter";
	var scriptVersion = "0.0.1";
	var scriptAuthor = "Nilas";
	var scriptURL = "https://github.com/NevermindNilas/TheAnimeScripter"

	//DEFAULT VALUES
	var defaultUpscaler = "ShuffleCugan";
	var defaultNrThreads = 2;
	var defaultCugan = "no-denoise";
	var defaultSwinIR = "small";
	var defaultSegment = "isnet-anime";
	var defaultInterpolateInt = 2;
	var defaultUpscaleInt = 2;

	var dialog = (panelGlobal instanceof Panel) ? panelGlobal : new Window("palette", undefined, undefined, {resizeable: true}); 
	    if ( !(panelGlobal instanceof Panel) ) dialog.text = "AnimeScripter by Nilas"; 
	    dialog.orientation = "column"; 
	    dialog.alignChildren = ["center","top"]; 
	    dialog.spacing = 20; 
	    dialog.margins = 20; 
	
	var divider1 = dialog.add("panel", undefined, undefined, {name: "divider1"}); 
	    divider1.alignment = "fill"; 
	
	// INTERPOLATEUPSCALEGROUP
	// =======================
	var InterpolateUpscaleGroup = dialog.add("group", undefined, {name: "InterpolateUpscaleGroup"}); 
	    InterpolateUpscaleGroup.orientation = "row"; 
	    InterpolateUpscaleGroup.alignChildren = ["left","center"]; 
	    InterpolateUpscaleGroup.spacing = 10; 
	    InterpolateUpscaleGroup.margins = 0; 
	
	// INTERPOLATEDPANEL
	// =================
	var InterpolatedPanel = InterpolateUpscaleGroup.add("panel", undefined, undefined, {name: "InterpolatedPanel"}); 
	    InterpolatedPanel.text = "Interpolate"; 
	    InterpolatedPanel.preferredSize.width = 175; 
	    InterpolatedPanel.orientation = "column"; 
	    InterpolatedPanel.alignChildren = ["left","top"]; 
	    InterpolatedPanel.spacing = 10; 
	    InterpolatedPanel.margins = 10; 
	
	// INTERPOLATEGROUP
	// ================
	var InterpolateGroup = InterpolatedPanel.add("group", undefined, {name: "InterpolateGroup"}); 
	    InterpolateGroup.orientation = "row"; 
	    InterpolateGroup.alignChildren = ["left","center"]; 
	    InterpolateGroup.spacing = 10; 
	    InterpolateGroup.margins = 0; 
	
	var InterpolateButton = InterpolateGroup.add("button", undefined, undefined, {name: "InterpolateButton"}); 
	    InterpolateButton.text = "Start"; 
	    InterpolateButton.preferredSize.width = 66; 
	
	var InterpolateInt = InterpolateGroup.add('edittext {justify: "center", properties: {name: "InterpolateInt"}}'); 
		InterpolateInt.text = app.settings.haveSetting("AnimeScripter", "InterpolateInt") ? app.settings.getSetting("AnimeScripter", "InterpolateInt") : "2"; 
		
	    InterpolateInt.preferredSize.width = 72; 
	
	// UPSCALEPANEL
	// ============
	var UpscalePanel = InterpolateUpscaleGroup.add("panel", undefined, undefined, {name: "UpscalePanel"}); 
	    UpscalePanel.text = "Upscale"; 
	    UpscalePanel.preferredSize.width = 175; 
	    UpscalePanel.orientation = "column"; 
	    UpscalePanel.alignChildren = ["left","top"]; 
	    UpscalePanel.spacing = 10; 
	    UpscalePanel.margins = 10; 
	
	// UPSCALEGROUP
	// ============
	var UpscaleGroup = UpscalePanel.add("group", undefined, {name: "UpscaleGroup"}); 
	    UpscaleGroup.orientation = "row"; 
	    UpscaleGroup.alignChildren = ["left","center"]; 
	    UpscaleGroup.spacing = 10; 
	    UpscaleGroup.margins = 0; 
	
	var UpscaleButton = UpscaleGroup.add("button", undefined, undefined, {name: "UpscaleButton"}); 
	    UpscaleButton.text = "Start"; 
	    UpscaleButton.preferredSize.width = 66; 
	
	var UpscaleInt = UpscaleGroup.add('edittext {justify: "center", properties: {name: "UpscaleInt"}}'); 
		UpscaleInt.text = app.settings.haveSetting("AnimeScripter", "UpscaleInt") ? app.settings.getSetting("AnimeScripter", "UpscaleInt") : "2"; 
		
	    UpscaleInt.preferredSize.width = 72; 
	
	// EXTRAANDFIRSTTIME
	// =================
	var ExtraAndFirstTime = dialog.add("group", undefined, {name: "ExtraAndFirstTime"}); 
	    ExtraAndFirstTime.orientation = "row"; 
	    ExtraAndFirstTime.alignChildren = ["left","center"]; 
	    ExtraAndFirstTime.spacing = 10; 
	    ExtraAndFirstTime.margins = 0; 
	
	// EXTRA
	// =====
	var Extra = ExtraAndFirstTime.add("panel", undefined, undefined, {name: "Extra"}); 
	    Extra.text = "Extra"; 
	    Extra.orientation = "column"; 
	    Extra.alignChildren = ["left","top"]; 
	    Extra.spacing = 10; 
	    Extra.margins = 10; 
	
	// DEDUPSEGMENTGROUP
	// =================
	var DedupSegmentGroup = Extra.add("group", undefined, {name: "DedupSegmentGroup"}); 
	    DedupSegmentGroup.orientation = "row"; 
	    DedupSegmentGroup.alignChildren = ["left","center"]; 
	    DedupSegmentGroup.spacing = 10; 
	    DedupSegmentGroup.margins = 0; 
	
	var DedupButton = DedupSegmentGroup.add("button", undefined, undefined, {name: "DedupButton"}); 
	    DedupButton.text = "Dedup"; 
	
	var SegmentButton = DedupSegmentGroup.add("button", undefined, undefined, {name: "SegmentButton"}); 
	    SegmentButton.text = "Segment"; 
	
	// ONFIRSTRUNONLY
	// ==============
	var OnFirstRunOnly = ExtraAndFirstTime.add("panel", undefined, undefined, {name: "OnFirstRunOnly"}); 
	    OnFirstRunOnly.text = "On first run only"; 
	    OnFirstRunOnly.orientation = "column"; 
	    OnFirstRunOnly.alignChildren = ["left","top"]; 
	    OnFirstRunOnly.spacing = 10; 
	    OnFirstRunOnly.margins = 10; 
	
	// GROUP1
	// ======
	var group1 = OnFirstRunOnly.add("group", undefined, {name: "group1"}); 
	    group1.orientation = "row"; 
	    group1.alignChildren = ["left","center"]; 
	    group1.spacing = 10; 
	    group1.margins = 0; 
	
	var OutputButton = group1.add("button", undefined, undefined, {name: "OutputButton"}); 
	    OutputButton.text = "Output"; 
	
	var MainpyButton = group1.add("button", undefined, undefined, {name: "MainpyButton"}); 
	    MainpyButton.text = "Main.py"; 
	
	// DIALOG
	// ======
	var divider1 = dialog.add("panel", undefined, undefined, {name: "divider1"}); 
	    divider1.alignment = "fill"; 
	
	// SETTINGSPANEL
	// =============
	var SettingsPanel = dialog.add("panel", undefined, undefined, {name: "SettingsPanel"}); 
	    SettingsPanel.text = "Settings"; 
	    SettingsPanel.orientation = "column"; 
	    SettingsPanel.alignChildren = ["left","top"]; 
	    SettingsPanel.spacing = 10; 
	    SettingsPanel.margins = 10; 
	
	// GROUP2
	// ======
	var group2 = SettingsPanel.add("group", undefined, {name: "group2"}); 
	    group2.orientation = "row"; 
	    group2.alignChildren = ["left","center"]; 
	    group2.spacing = 10; 
	    group2.margins = 0; 
	
	var divider2 = group2.add("panel", undefined, undefined, {name: "divider2"}); 
	    divider2.alignment = "fill"; 
	
	var ChooseUpscalerText = group2.add("statictext", undefined, undefined, {name: "ChooseUpscalerText"}); 
	    ChooseUpscalerText.text = "Upscaler"; 
	    ChooseUpscalerText.preferredSize.width = 61; 
	    ChooseUpscalerText.justify = "center"; 
	
	var DropdownUpscaler_array = ["ShuffleCugan","-","Cugan","-","UltraCompact","-","Compact","-","SwinIR"]; 
	var DropdownUpscaler = group2.add("dropdownlist", undefined, undefined, {name: "DropdownUpscaler", items: DropdownUpscaler_array}); 
		DropdownUpscaler.selection = app.settings.haveSetting("AnimeScripter", "DropdownUpscaler") ? app.settings.getSetting("AnimeScripter", "DropdownUpscaler") : 0; 
	
	
	var divider3 = group2.add("panel", undefined, undefined, {name: "divider3"}); 
	    divider3.alignment = "fill"; 
	
	var ChooseNrThreads = group2.add("statictext", undefined, undefined, {name: "ChooseNrThreads"}); 
	    ChooseNrThreads.text = "Nr of Threads"; 
	
	var NumberOfThreadsInt = group2.add('edittext {justify: "center", properties: {name: "NumberOfThreadsInt"}}'); 
		NumberOfThreadsInt.text = app.settings.haveSetting("AnimeScripter", "NumberOfThreadsInt") ? app.settings.getSetting("AnimeScripter", "NumberOfThreadsInt") : "2"; 		
	    NumberOfThreadsInt.preferredSize.width = 73; 
	
	// GROUP3
	// ======
	var group3 = SettingsPanel.add("group", undefined, {name: "group3"}); 
	    group3.orientation = "row"; 
	    group3.alignChildren = ["left","center"]; 
	    group3.spacing = 10; 
	    group3.margins = 0; 
	
	var divider4 = group3.add("panel", undefined, undefined, {name: "divider4"}); 
	    divider4.alignment = "fill"; 
	
	var ChooseCuganText = group3.add("statictext", undefined, undefined, {name: "ChooseCuganText"}); 
	    ChooseCuganText.text = "Cugan"; 
	    ChooseCuganText.preferredSize.width = 62; 
	    ChooseCuganText.justify = "center"; 
	
	var DropdownCugan_array = ["no-denoise","-","conservative","-","denoise1x","-","denoise2x",""]; 
	var DropdownCugan = group3.add("dropdownlist", undefined, undefined, {name: "DropdownCugan", items: DropdownCugan_array}); 
		DropdownCugan.selection = app.settings.haveSetting("AnimeScripter", "DropdownCugan") ? app.settings.getSetting("AnimeScripter", "DropdownCugan") : 0; 
	
	
	var divider5 = group3.add("panel", undefined, undefined, {name: "divider5"}); 
	    divider5.alignment = "fill"; 
	
	var ChooseSwinIRText = group3.add("statictext", undefined, undefined, {name: "ChooseSwinIRText"}); 
	    ChooseSwinIRText.text = "SwinIR"; 
	    ChooseSwinIRText.preferredSize.width = 85; 
	    ChooseSwinIRText.justify = "center"; 
	
	var DropdownSwinIr_array = ["small","-","medium","-","large"]; 
	var DropdownSwinIr = group3.add("dropdownlist", undefined, undefined, {name: "DropdownSwinIr", items: DropdownSwinIr_array}); 
		DropdownSwinIr.selection = app.settings.haveSetting("AnimeScripter", "DropdownSwinIr") ? app.settings.getSetting("AnimeScripter", "DropdownSwinIr") : 0; 
	    DropdownSwinIr.preferredSize.width = 71; 
	
	// GROUP4
	// ======
	var group4 = SettingsPanel.add("group", undefined, {name: "group4"}); 
	    group4.orientation = "row"; 
	    group4.alignChildren = ["left","center"]; 
	    group4.spacing = 10; 
	    group4.margins = 0; 
	
	var divider6 = group4.add("panel", undefined, undefined, {name: "divider6"}); 
	    divider6.alignment = "fill"; 
	
	var ChooseSegmentText = group4.add("statictext", undefined, undefined, {name: "ChooseSegmentText"}); 
	    ChooseSegmentText.text = "Segment"; 
	    ChooseSegmentText.preferredSize.width = 63; 
	    ChooseSegmentText.justify = "center"; 
	
	var DropdownSegment_array = ["isnet-anime","-","isnet-general","-",""]; 
	var DropdownSegment = group4.add("dropdownlist", undefined, undefined, {name: "DropdownSegment", items: DropdownSegment_array}); 
		DropdownSegment.selection = app.settings.haveSetting("AnimeScripter", "DropdownSegment") ? app.settings.getSetting("AnimeScripter", "DropdownSegment") : 0; 
	    DropdownSegment.enabled = false; 
	    DropdownSegment.preferredSize.width = 72; 

	InterpolateButton.onClick = function() {
		app.settings.saveSetting("AnimeScripter", "InterpolateInt", InterpolateInt.text);
		process('interpolate');
	}
	
	UpscaleButton.onClick = function() {
		app.settings.saveSetting("AnimeScripter", "UpscaleInt", UpscaleInt.text);
		process('upscale');
	}
	
	DedupButton.onClick = function() {
		process('dedup');
	}

	SegmentButton.onClick = function() {
		process('segment');
	}

	OutputButton.onClick = function() {
		try {
			var folder = new Folder()
			var outputFolder = folder.selectDlg("Select an output directory");
			if (outputFolder != null) {
				app.settings.saveSetting("AnimeScripter", "outputDirectory", outputFolder.fsName);
			}
			alert("successfully saved path");
		} catch (error) {
			alert(error);
		}
	}

	MainpyButton.onClick = function() {
		try {
			var mainPyFile = File.openDialog("Select the main.py file");
			if (mainPyFile != null) {
				app.settings.saveSetting("AnimeScripter", "mainPyPath", mainPyFile.fsName);
			}
			alert("successfully saved path");
		} catch (error) {
			alert(error);
		}
	}
	InterpolateInt.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "InterpolateInt", InterpolateInt.text);
	}

	DropdownUpscaler.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "DropdownUpscaler", DropdownUpscaler.selection.index);
	}
	
	NumberOfThreadsInt.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "NumberOfThreadsInt", NumberOfThreadsInt.text);
	}
	
	DropdownSwinIr.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "DropdownSwinIr", DropdownSwinIr.selection.index);
	}
	
	DropdownCugan.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "DropdownCugan", DropdownCugan.selection.index);
	}
	
	DropdownSegment.onChange = function() {
		app.settings.saveSetting("AnimeScripter", "DropdownSegment", DropdownSegment.selection.index);
	}
	
	function process(module) {
		var outputFolder = app.settings.haveSetting("AnimeScripter", "outputDirectory") ? app.settings.getSetting("AnimeScripter", "outputDirectory") : "";
		var mainPyFile = app.settings.haveSetting("AnimeScripter", "mainPyPath") ? app.settings.getSetting("AnimeScripter", "mainPyPath") : "";
		var DropdownCugan = app.settings.haveSetting("AnimeScripter", "DropdownCugan") ? app.settings.getSetting("AnimeScripter", "DropdownCugan") : defaultCugan;
		var DropdownSwinIr = app.settings.haveSetting("AnimeScripter", "DropdownSwinIr") ? app.settings.getSetting("AnimeScripter", "DropdownSwinIr") : defaultSwinIR;
		var DropdownSegment = app.settings.haveSetting("AnimeScripter", "DropdownSegment") ? app.settings.getSetting("AnimeScripter", "DropdownSegment") : defaultSegment;
		var DropdownUpscaler = app.settings.haveSetting("AnimeScripter", "DropdownUpscaler") ? app.settings.getSetting("AnimeScripter", "DropdownUpscaler") : defaultUpscaler;
		var NumberOfThreadsInt = app.settings.haveSetting("AnimeScripter", "NumberOfThreadsInt") ? app.settings.getSetting("AnimeScripter", "NumberOfThreadsInt") : defaultNrThreads;
		var InterpolateInt = app.settings.haveSetting("AnimeScripter", "InterpolateInt") ? app.settings.getSetting("AnimeScripter", "InterpolateInt") : defaultInterpolateInt;
		var UpscaleInt = app.settings.haveSetting("AnimeScripter", "UpscaleInt") ? app.settings.getSetting("AnimeScripter", "UpscaleInt") : defaultUpscaleInt;
		
		if (((!app.project) || (!app.project.activeItem)) || (app.project.activeItem.selectedLayers.length != 1)) {
			return alert("Please select one layer.");
		}

		if (outputFolder == "" || outputFolder == null) {
			alert("No output directory selected");
			return;
		}
	
		if (mainPyFile == "" || mainPyFile == null) {
			alert("No main.py file selected");
			return;
		}

		var comp = app.project.activeItem;
		var layer = comp.selectedLayers[0];
		var activeLayerPath = layer.source.file.fsName;
		var activeLayerName = layer.name.replace('.mp4', '');
		
		var pyfile = File(mainPyFile);
		var scriptPath = pyfile.parent.fsName;
				

		/*var inPoint = layer.inPoint;
		var outPoint = layer.outPoint;

		var duration = outPoint - inPoint;

		if (duration !== layer.source.duration) {
			output_name = outputFolder + "\\" + activeLayerName + "_trimmed" + ".mp4";
			alert(inPoint)
			alert(outPoint)
			command = "cmd.exe /c" + " ffmpeg -i " + activeLayerPath + " -ss " + inPoint + " -to " + outPoint + "-vcodec copy" + output_name + "-y";
			var result = system.callSystem(command);
			activeLayerPath = output_name;
		} else {
		}
		*/

		output_name = outputFolder + "\\" + activeLayerName + "_" + module + ".mp4";
		var command = "";
		if (module == "interpolate") {
			command = "cd " + scriptPath + " && python " + mainPyFile + " -video " + activeLayerPath + " -model_type rife -multi " + InterpolateInt + " -output " + output_name;
		} else if (module == "upscale") {
			if (DropdownUpscaler == "ShuffleCugan") {
				command = "cd " + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type shufflecugan -output " + output_name + " -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt;
			} else if (DropdownUpscaler == "Cugan") {
				command = "cd " + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type cugan -output " + output_name + " -nt " + NumberOfThreadsInt + " -kind_model " + DropdownCugan + " -multi " + UpscaleInt;
			} else if (DropdownUpscaler == "UltraCompact") {
				command = "cd " + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type ultracompact -output " + output_name + " -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt;
			} else if (DropdownUpscaler == "Compact") {
				command = "cd " + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type compact -output " + output_name + " -nt " + NumberOfThreadsInt + " -multi " + UpscaleInt;
			} else if (DropdownUpscaler == "Swwinir") {
				command = "cd " + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type swinir -output " + output_name + " -nt " + NumberOfThreadsInt + " -kind_model " + DropdownSwinIr + " -multi " + UpscaleInt;
			}
		} else if (module == "dedup") {
			command = "cd" + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type dedup -output " + output_name + " -kind_model " + "ffmpeg";
		} else if (module == "segment") {
			command = "cd" + scriptPath + "&& python " + mainPyFile + " -video " + activeLayerPath + " -model_type segment -output " + output_name + "-kind_model " + DropdownSegment;
		} else {
			alert("Something went wrong");
			return;
		}
		if (layer) {
			try {
				var cmdCommand = 'cmd.exe /c "' + command + '"';
				var result = system.callSystem(cmdCommand);
				var importOptions = new ImportOptions(File(output_name));
				var importedFile = app.project.importFile(importOptions);
				var inputLayer = comp.layers.add(importedFile);
				inputLayer.moveBefore(layer);
			} catch (error) {
				alert(error);
			}
		}
	}
	dialog.layout.layout(true);
	dialog.layout.resize();
	dialog.onResizing = dialog.onResize = function () { this.layout.resize(); }
  
	if ( dialog instanceof Window ) dialog.show();
  
	return dialog;
  
  }());