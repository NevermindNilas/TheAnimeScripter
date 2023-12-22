# jsxbin

Convert jsx ExtendScript files into jsxbin

## Example

```javascript
const jsxbin = require( 'jsxbin' )

jsxbin( 'path/to/script.js', 'output/script.jsxbin' )
	.then( outputfiles => {
		console.log( 'Finished!' )
	})
	.catch( err => {
		console.error( err )
	})
```

## Methods

### jsxbin( inputPaths, [outputPath] )

`inputPaths` can be:

- String with path to jsx file. `script.jsx`
- String with glob pattern that matches jsx/js files. `*.jsx`
- Array of any of the above

`outputPath`, optional, can be:

- String path to converted file. `path/to/script.jsxbin`
	- Should only be used when passing only one file as `inputPaths`
- String path to converted file directory. `path/to/output`
- Array of string paths of names for all converted files
	- Should only be used when passing an array to `inputPaths`. Input and output arrays must be the same length.
- If not given, the files will be created in the same directory as the input file(s)

`jsxbin` returns a promise with an array of file paths to the converted files

### Examples

```javascript
// Just one file
jsxbin( 'script.jsx', 'script.jsxbin' )

// Is the same as
jsxbin( 'script.jsx' )

// Multiple files
jsxbin([ 'script1.jsx', 'script2.jsx' ], 'output/' )

// Using glob string for input, jsxbin files will be placed in 'output/' dir
jsxbin( 'src/*.jsx', 'output' )

// With no output path specified, jsxbin files will be placed in the 'src/' dir
jsxbin( 'src/*jsx' )

// As a gulp task
gulp.task( 'jsxbin', () => {
	return jsxbin( 'src/index.js', 'output/script.jsxbin' )
})
```

## From the Command Line

This package also includes a `jsxbin` command than can be run from the command line.

```
jsxbin

  usage: jsxbin -i file1.jsx, file2.jsx.. -o outputdir
  usage: jsxbin -i file1.jsx -o outputname.jsxbin

  Converts Extendscript .jsx files into jsxbin

Options

  -i, --input   file(s)      The file or files to convert
  -o, --output  file|folder  The file or folder where the converted file will be placed
  -v, --verbose              Show more info while running
  --debug                    Show even more info while running
  -h, --help                 Show help
```

## Install

with npm do:

```
npm install jsxbin
```
to get the function, or

```
npm install jsxbin -g
```
to get the command.

## Contributing

Issues and pull requests are more than welcome! Please ensure you have tests for your pull requests, and that `npm test` passes.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
This project includes code from the [Adobe Extendscript debugger extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=Adobe.extendscript-debug) which uses the [Apache License, Version 2.0]*(http://www.apache.org/licenses/LICENSE-2.0)

## Thanks

- The implemention using VSCode ES debugger from Adobe was added by [Sammarks](https://github.com/sammarks)
- Thanks to [RenderTom](https://github.com/rendertom) and [Zlovatt](https://github.com/zlovatt) for additional contributions
