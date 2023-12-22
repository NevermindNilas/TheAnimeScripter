#!/usr/bin/env node

const jsxbin = require( './index' )
const cmdArgs = require( 'command-line-args' )
const log = require( './src/logger' )

// Called from command line
const optionDefinitions = [
	{ name: 'input', alias: 'i', type: String, multiple: true },
	{ name: 'output', alias: 'o', type: String },
	{ name: 'verbose', alias: 'v', type: Boolean },
	{ name: 'debug', type: Boolean },
	{ name: 'help', alias: 'h', type: Boolean }
]

const options = cmdArgs( optionDefinitions, { partial: true })

if ( options.verbose ) {
	log.level = 'verbose'
}
if ( options.debug ) {
	log.level = 'debug'
}

log.debug( 'options', { options })

if ( options.help ) {
	showUsage()
} else {
	if ( !options.input || !options.output ) {
		showUsage()
	}
	jsxbin( options.input, options.output )
		.catch( log.error )
}

function showUsage() {
	const getUsage = require( 'command-line-usage' )
	const sections = [
		{
			header: 'jsxbin',
			content: 'Converts Extendscript .jsx files into jsxbin files using ExtendScript Toolkit'
		},
		{
			header: 'Options',
			optionList: [
				{
					name: 'input',
					alias: 'i',
					typeLabel: '{underline file(s)}',
					description: 'The file or files to convert'
				},
				{
					name: 'output',
					alias: 'o',
					typeLabel: '{underline file}|{underline folder}',
					description: 'The file or folder where the converted file will be placed'
				},
				{
					name: 'verbose',
					alias: 'v',
					description: 'Show more info while running'
				},
				{
					name: 'debug',
					description: 'Show even more info while running'
				},
				{
					name: 'help',
					alias: 'h',
					description: 'Show this help'
				}
			]
		}
	]
	console.log( getUsage( sections ) )
}
