const exec = require( 'child_process' ).exec

const archive = `esdebugger-core-${process.platform === 'darwin' ? 'mac' : 'win'}.tar.gz`
exec( `tar -xzf ${archive}` )
