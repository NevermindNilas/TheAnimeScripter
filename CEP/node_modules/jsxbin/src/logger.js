const winston = require( 'winston' )
const logger = winston.createLogger()

logger.add( new winston.transports.Console({
	format: winston.format.combine(
		winston.format.padLevels(),
		winston.format.colorize(),
		winston.format.simple()
	)
}) )

module.exports = logger
