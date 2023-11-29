// Ae Script
// I don't know how to code in js, so this is a little messy
aeScript()

function aeScript() {
    var upscaleModel = "ShuffleCugan"
    var numberOfThreads = 2
    var cuganModel = "no-denoise"
    var swinIRModel = "small"
    var segmentModel = "isnet-anime"
    
    newPanel();

    function newPanel() {
        var win = new Window('palette', 'Anime Scripter', undefined, { resizeable: true });

        var settingsButton = win.add('button', undefined, 'Settings');
        settingsButton.characters = 20;

        var interpolationGroup = win.add('group');
        interpolationGroup.orientation = 'row';

        var interpolationButton = interpolationGroup.add('button', undefined, 'Interpolation');
        interpolationButton.characters = 13;

        var inputBox = interpolationGroup.add('edittext', undefined, '2');
        inputBox.characters = 5;

        var upscaleGroup = win.add('group');
        upscaleGroup.orientation = 'row';
        
        var upscaleButton = upscaleGroup.add('button', undefined, 'Upscale');
        upscaleButton.characters = 13;

        var inputBox = upscaleGroup.add('edittext', undefined, '2');
        inputBox.characters = 5;

        var dedupSegmentGroup = win.add('group');
        dedupSegmentGroup.orientation = 'row';
        
        var dedupButton = dedupSegmentGroup.add('button', undefined, 'Dedup');
        dedupButton.characters = 8;

        var segmentButton = dedupSegmentGroup.add('button', undefined, 'Segment');
        segmentButton.characters = 8;
        
        settingsButton.onClick = function() {
            var winSettings = new Window('dialog', 'Settings', undefined)

            var mainPyButton = winSettings.add('button', undefined, 'Select main.py');
            mainPyButton.characters = 20;
    
            var mainPyPath;
    
            mainPyButton.onClick = function() {
                var mainPyFile = File.openDialog("Select main.py");
                if (mainPyFile) {
                    mainPyPath = mainPyFile.fsName;
                    alert("Selected file: " + mainPyPath);
                }
            }

            var outputPathButton = winSettings.add('button', undefined, 'Select Output Path');
            outputPathButton.characters = 20;

            var outputPath;

            outputPathButton.onClick = function() {
                var outputPathFile = Folder.selectDialog("Select Output Folder");
                if (outputPathFile) {
                    outputPath = outputPathFile.fsName;
                    alert("Selected output path: " + outputPath);
                }
            }
            
            var upscaleSettings = winSettings.add('group')
            upscaleSettings.orientation = 'row'

            var upscaleButton = upscaleSettings.add('button', undefined, 'Upscaler')
            upscaleButton.characters = 15

            var upscaleDropdown = upscaleSettings.add('dropdownlist', undefined, ["ShuffleCugan", "Cugan", "Compact", "Ultracompact", "SwinIR"])
            upscaleDropdown.selection = 0

            var numberOfThreads = winSettings.add('group')
            numberOfThreads.orientation = 'row'

            var numberOfThreadsButton = numberOfThreads.add('button', undefined, 'Number of Threads')
            numberOfThreadsButton.characters = 15

            var numberOfThreadsText = numberOfThreads.add('edittext', undefined, '2')
            numberOfThreadsText.characters = 10

            var cuganSettings = winSettings.add('group')
            cuganSettings.orientation = 'row'

            var cuganButton = cuganSettings.add('button', undefined, 'Cugan Model')
            cuganButton.characters = 15

            var cuganDropdown = cuganSettings.add('dropdownlist', undefined, ["no-denoise", "conservative", "denoise2x", "denoise3x"])
            cuganDropdown.selection = 0

            var swinIRSettings = winSettings.add('group')
            swinIRSettings.orientation = 'row'

            var swinIRButton = swinIRSettings.add('button', undefined, 'SwinIR Model')
            swinIRButton.characters = 15

            var swinIRDropdown = swinIRSettings.add('dropdownlist', undefined, ["small", "medium", "large"])
            swinIRDropdown.selection = 0

            var segmentSettings = winSettings.add('group')
            segmentSettings.orientation = 'row'

            var segmentButton = segmentSettings.add('button', undefined, 'Segment Model')
            segmentButton.characters = 15

            var segmentDropdown = segmentSettings.add('dropdownlist', undefined, ["isnet-anime", "isnet-general-purpose"])
            segmentDropdown.selection = 0

            upscaleDropdown.onChange = function() {
                upscaleModel = this.selection.text;
            }
            
            numberOfThreadsText.onChange = function() {
                numberOfThreads = parseInt(this.text);
            }

            cuganDropdown.onChange = function() {
                cuganModel = this.selection.text;
            }

            swinIRDropdown.onChange = function() {
                swinIRModel = this.selection.text;
            }
            
            segmentDropdown.onChange = function() {
                segmentModel = this.selection.text;
            }
    
            var tip1 = winSettings.add('statictext', undefined, "- Don't go above 4 threads");
            var tip2 = winSettings.add('statictext', undefined, "- Models are ordered by speed");
            var tip3 = winSettings.add('statictext', undefined, "- More info at https://github.com/NevermindNilas/TheAnimeScripter/");
            tip1.characters = 20
            tip2.characters = 20
            tip3.characters = 40

            tip1.justify = 'center'
            tip2.justify = 'center'
            tip3.justify = 'center'
            //var cuganProCheckbox = winSettings.add('checkbox', undefined, 'Cugan Pro');

            winSettings.show();
        }
        interpolationButton.onClick = function() {
            process();
        }
        
        function process() {
            var comp = app.project.activeItem;
            if (comp) {
                var videoPath = comp.file.fsName;
                var command = 'python "' + mainPyPath + '" -video "' + videoPath + '" -model rife -multi ' + numberOfThreads;
                $.writeln("Running command: " + command); // For debugging
                var result = $.system(command);
                $.writeln("Command result: " + result); // For debugging

                // Assuming the output file has the same name as the input file but with '_output' appended
                var outputFilePath = outputPath + "/" + comp.file.name.replace(/\.[^\.]+$/, "") + "_output.mp4";

                // Check if the output file exists
                var outputFile = new File(outputFilePath);
                if (outputFile.exists) {
                    // Import the output file into the After Effects project
                    var importedFile = app.project.importFile(new ImportOptions(outputFile));
                    $.writeln("Imported file: " + importedFile.name); // For debugging
                } else {
                    alert("Output file does not exist: " + outputFilePath);
                }
            } else {
                alert("No active item selected");
            }
        }

        var addText = win.add('statictext', undefined, 'youtube.com/@NilasEdits');
        addText.characters = 20;
        addText.justify = 'center'; // Center the text

        win.onResizing = win.onResize = function () {
            this.layout.resize();
        };

        win instanceof Window
            ? (win.center(), win.show()) : (win.layout.layout(true), win.layout.resize());
    
    }
}