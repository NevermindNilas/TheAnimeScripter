#!/usr/bin/env node

// Require modules
const path = require( 'path' )
const glob = require( 'glob' )

const log = require( './src/logger' )
const createDir = require( './src/createDir' )
const convertScripts = require( './src/convertScripts' )

log.level = '1'

module.exports = jsxbin
module.exports.getInputPaths = getInputPaths
module.exports.getOutputPaths = getOutputPaths

/**
 * Converts input file into jsxbin file using ExtendScript Toolkit
 * @function jsxbin
 * @param  {string|string[]|object} inputPaths  The file or files to convert, can be in
 *         glob paths (``*.jsx`) or regular paths that point to files,
 *         not directories (`/path/to/script.jsx`)
 * @param  {string|string[]} [outputPath] The output file or output directory,
 *         or an array of output files. If not given, the files will be created
 *         in the same location as the input file(s)
 * @return {Promise} A Promise that returns an array of file paths to the
 *         converted files
 */
function jsxbin( inputPaths, outputPath ) {
	if ( !Array.isArray( inputPaths ) && typeof inputPaths === 'object' ) {
		const options = inputPaths
		inputPaths = options.input
		outputPath = options.output
	}

	// Debug some values
	log.debug( `Current dir: ${process.cwd()}` )
	log.debug( 'arguments', { inputPaths, outputPath })

	// Store input and output globaly, because they need to be accesible later
	let input, output

	// "inputPaths" can be different things, so we need to convert it to the
	// correct value, an array of absolute paths, that can be used in the
	// ESTK script.
	return getInputPaths( inputPaths )
		.then( inputPaths => {
			input = inputPaths

			// We also have to convert outputPath into an array of absolute paths
			output = getOutputPaths( input, outputPath )
			if ( outputPath === undefined ) {
				outputPath = output[0]
			}
		})

	// We have to create the output folder if it does not exist
		.then( () => createDir( outputPath ) )

	// Convert the script using the resources from the VSCode extension.
		.then( () => convertScripts( input, output ) )
		.then( () => {
			log.info( 'Finished!' )
			return output
		})
}

function getInputPaths( inputPaths ) {
	// We are going to loop through all input paths, so make sure it is an array
	if ( !Array.isArray( inputPaths ) ) {
		inputPaths = [ inputPaths ]
	}

	// We are using glob to convert any pattern strings into asolute paths
	const globOptions = {

		// We do not want any folders to show up in the match, only files
		nodir: true,

		// All paths should be absolute, because the script in ESTK will not be
		// executed from the same place as the converted files are located
		absolute: true

	}

	const allPaths = []

	// Glob is async, and we are looping, so we need to store all promises and
	// wait for them to finish
	const globPromises = []
	inputPaths.forEach( pattern => {
		const promise = new Promise( ( resolve, reject ) => {
			glob( pattern, globOptions, ( err, paths ) => {
				if ( err ) {
					return reject( err )
				}

				// Promise.all merges all promise values into an array,
				// paths is an array, and that would make a 2D array, which we
				// do not want. So push path values to a different array instead
				allPaths.push.apply( allPaths, paths )
				resolve()
			})
		})
		globPromises.push( promise )
	})

	// Wait for all glob paths to finish, then return all the paths
	return Promise.all( globPromises ).then( () => allPaths )
}

function getOutputPaths( inputPaths, outputPath ) {
	const output = []

	if ( Array.isArray( outputPath ) ) {
		if ( outputPath.length !== inputPaths.length ) {
			throw new Error( 'jsxbin error: When passing an array as output it must have the same length as number of files in input' )
		}
		return outputPath
	}

	if ( outputPath === undefined ) {
		return inputPaths.map( filePath => {
			const extension = path.extname( filePath )
			return filePath.replace( extension, '.jsxbin' )
		})
	}

	// Check if outputPath is directory
	if ( /\.jsxbin$/.test( outputPath ) ) {
		// "outputPath" is a file
		// We need output to be an array with the same length of the input
		inputPaths.forEach( () => {
			// FIXME: this will cause all output files to have the same name
			// if there are multiple input files
			output.push( outputPath )
		})

		// "outputPath" is a directory

		inputPaths.forEach( filePath => {
			// Replace the extension of the filename with jsxbin and put it
			// in the output directory
			const fileName = replaceExtension( filePath, 'jsxbin' )
			output.push( path.join( outputPath, fileName ) )
		})

	// "outputPath" is a directory
	} else {
		inputPaths.forEach( filePath => {
			// Replace the extension of the filename with jsxbin and put it
			// in the output directory
			const fileName = replaceExtension( filePath, 'jsxbin' )
			output.push( path.join( outputPath, fileName ) )
		})
	}
	return output
}

function replaceExtension( filePath, newExtension ) {
	const extension = path.extname( filePath )
	const fileName = path.basename( filePath, extension )

	return `${fileName}.${newExtension}`
}
