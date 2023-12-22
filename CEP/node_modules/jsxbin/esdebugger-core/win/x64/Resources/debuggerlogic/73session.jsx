/************************************************************************
*
*  @@@BUILDINFO@@@ 73session-2.jsx 3.5.0.48	14-December-2009
*  ADOBE SYSTEMS INCORPORATED
*  Copyright 2010 Adobe Systems Incorporated
*  All Rights Reserved.
* 
* NOTICE:  Adobe permits you to use,  modify, and  distribute this file in
* accordance with the terms of the Adobe license agreement accompanying it.
* If you have received this file from a source other than Adobe, then your
* use, modification, or distribution of it requires the prior written
* permission of Adobe.
*
**************************************************************************/

//
// current active session
//
DebugSession.current = null;

//
// active sessions
//
DebugSession.activeCount = 0;

DebugSession.addActiveSession = function()
{		
	DebugSession.activeCount++;
}

DebugSession.removeActiveSession = function()
{
	DebugSession.activeCount--;

	if( DebugSession.activeCount < 0 )
		DebugSession.activeCount = 0;
}

//
// session state constants
//  
DebugSession.RUNNING			= "running";
DebugSession.STOPPED			= "stopped";
DebugSession.INACTIVE			= "";

//
// debugging commands
//
DebugSession.CMD_CONTINUE       = "continue";
DebugSession.CMD_PAUSE          = "pause";
DebugSession.CMD_STEPOVER       = "step over";
DebugSession.CMD_STEPINTO       = "step into";
DebugSession.CMD_STEPOUT        = "step out";

//
// error strings
//
const kMsgHaltedExecution = "Execution halted";

const kErrMsgEngineBusy	  = "ENGINE BUSY";

//
// error codes
//
const kErrEngineNotExists	= 57;
const kErrEngineHalted		= -34;
const kErrBadAction			= 32;

//
// max restarts
//
DebugSession.kRestartSessionMAX = 3;

///////////////////////////////////////////////////////////////////////////////
//
// Debugger broadcaster
//
DebugSession.broadcaster = new Broadcaster;

DebugSession.registerClient = function( clientObj, msg )
{ return DebugSession.broadcaster.registerClient( clientObj, msg ); }

DebugSession.unregisterClient = function( clientObj, msg )
{ DebugSession.broadcaster.unregisterClient( clientObj, msg ); }

DebugSession.notifyClients = function( reason, param01, param02, param03, param04 )
{ DebugSession.broadcaster.notifyClients( reason, param01, param02, param03, param04 ); }

//-----------------------------------------------------------------------------
// 
// Session.prepareSession(...)
// 
// Purpose: Prepare to start a session
// 
//-----------------------------------------------------------------------------

DebugSession.prepareSession = function( target, session, dbgLevel, doc, saveDocs )
{

	if( !doc.includePath || ( doc.includePath && doc.includePath.length == 0 ) )
	{
	    var f = new File( doc.scriptID );
	    if( f.exists )
			doc.includePath = f.parent ? f.parent.absoluteURI : "/";
	}
    
  	var isSyntaxCorrect =  true;//doc.checkSyntax( doc.includePath );
  	var engineName = '';
        
    //
    // start session, create before if not available
    //
    var data = { engine : engineName, session : session, target : target, document : doc, debugLevel : dbgLevel, error : new ErrorInfo() };

    if( target.getConnected() )
    {
        log("[" + (new Date()).valueOf() + "] prepareSession Target connected. Starting session.");
        DebugSession.startSession( data );
    }
    else
	{
		log("[" + (new Date()).valueOf() + "] prepareSession Target is not connected. Trying to connect.");
		target.prepareDebugSession = true;
        target.connect( false, new Callback( DebugSession.startSession, data ), data.error );
	}
}

//-----------------------------------------------------------------------------
// 
// DebugSession.startSession(...)
// 
// Purpose: Start given session
// 
//-----------------------------------------------------------------------------

DebugSession.startSession = function( data )
{
    var session = data.session;
    //
    // get session from TargetInfo if not in data
    //
    if( !session && data.target )
    {
		if( data.engine )
			session = data.target.findSessionEngine( data.engine );
		else
			session = data.target.getActive();
		//
		// If session does not exists then add the session and 
		// give it a try. If the target response with error
		// "Engine does not exists" then the session will be
		// removed.
		//
        if( !session && data.engine )
		{
			var addr			= new Address( data.target.address );
			addr.engine			= data.engine;
            session				= data.target.addSession( addr );
			session.temporary	= true;
		}
			
		if( session )
			data.target.setActive( session, false, true );
			
        session = data.target.activeSession;
    }
    
    //
    // finaly start the session
    //
	if( data.target && appInstance.targetAppRunning( data.target ) )
	{
		if( session )
		{
			session.startInfo = null;
			session.startSession( data.document, data.debugLevel, session, data.error );
		}
		else
		{
			if( data.target )
				data.target.prepareDebugSession = false;

			try
			{
				if( data.error.length() <= 0 ){
					log("[" + (new Date()).valueOf() + "] startSession Error: Target " + data.target.getTitle() + " provides no engine for debugging.");
				}
				log("[" + (new Date()).valueOf() + "] startSession Error: Cannot execute script.");
			}
			catch( exc )
			{}
		}
	}
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: ctor
// 
//-----------------------------------------------------------------------------

function DebugSession( addr, target, sessionObj )
{
    this.address        = addr;                     		// session address
    this.target         = target;                   		// target of session
    this.sessionObj     = sessionObj;               		// cdi session object
    this.document       = null;                     		// current document in debug session
    this.documents      = {};                       		// documents of debug session
    this.dbgLevel       = ExecutionContext.DBGLEVEL_BREAK;	// current debug level for execution
    this.state          = DebugSession.INACTIVE;    		// current state of session
    this.line           = -1;                       		// current line of debugging
    this.frame          = -1;                       		// current active frame
    this.stack          = [];                       		// last known stack
    this.error          = null;                     		// last execution error message
	
	this.startInfo		= null;								// store related data for start debug session in order to restart the session if it failed

	this.temporary		= false;							// if true and the execution fails then this session will be removed

	this.releasing		= false;							// if true the sessionObj is about to be released. NO command should be sent to the engine!
    
    this.finalized      = false;                    		// session was finalized after execution
    this.finalizing     = false;                            // true if just finalizing the session after execution

	this.stopExe		= false;							// true when about to stop script execution
	this.stopTaskID		= -1;
    
    this.tmpSilentStop  = undefined;                        // true if the debugger should stop without any notification
    
	this.isInitialized	= false;							// DebugSession initialized?
    this.initializing	= true;								// about to be initialized

	DebugSession.registerClient( this , 'initialized' );
    
    if( this.sessionObj )
    {
        this.sessionObj.onTask      = this.processTasks;
        this.sessionObj.dbgSession  = this;
        this.initializing           = false;
            
        DebugSession.notifyClients( 'initialized', this.target, this, true );
    }
    else
    {
        if( this.target && this.target.cdic )
        {
            //
            // get cdi session object and connect
            //
            try
            {
                var job = this.target.cdic.acquireSession( this.address );
                job.dbgSession = this;
                
				if( !this.sessionObj && this.initializing )
				{
					//
					// we could setup the sessionObject right after
					// creating the job
					//
					this.sessionObj            = job.sessionObject;
					this.sessionObj.onTask     = this.processTasks;
					this.sessionObj.dbgSession = this;
				}

                job.onResult = function()
                {
                    var sessionObj = this.sessionObject;
                    
                    if( sessionObj )
                    {
                        if( this.dbgSession.initializing )
                        {
                            this.dbgSession.sessionObj            = sessionObj;
                            this.dbgSession.sessionObj.onTask     = this.dbgSession.processTasks;
                            this.dbgSession.sessionObj.dbgSession = this.dbgSession;

						    if( this.dbgSession.target.getFeature( Feature.CONNECT, this.dbgSession ) )
						    {
							    try
							    {           
								    var job = this.dbgSession.sessionObj.connect();
								    job.dbgSession = this.dbgSession;
    	                            
								    job.onResult = function()
								    {
									    this.dbgSession.initializing = false;	
											
									    DebugSession.notifyClients( 'initialized', this.dbgSession.target, this.dbgSession, true );
								    }
    	                            
								    job.onError =  job.onTimeout = function()
								    {
								    	log("[" + (new Date()).valueOf() + "] DebugSession Error: Connection Error. Releasing Session");
									    //
									    // error on connecting, release session
									    //
									    this.dbgSession.initializing = false;
									    this.dbgSession.release( true );
									    DebugSession.notifyClients( 'initialized', this.dbgSession.target, this.dbgSession, false );
								    }
    	                            
								    job.submit();
							    }
							    catch( exc )
							    {
							    	log("[" + (new Date()).valueOf() + "] DebugSession Exception: Connection Error. Releasing Session");
								    //
								    // error on connecting, release session
								    //
								    this.dbgSession.initializing = false;
								    this.dbgSession.release( true );
								    DebugSession.notifyClients( 'initialized', this.dbgSession.target, this.dbgSession, false );
							    }
						    }
						    else
						    {
							    this.dbgSession.initializing = false;
									
							    DebugSession.notifyClients( 'initialized', this.dbgSession.target, this.dbgSession, true );
						    }
                        }
                        else if( sessionObj != this.dbgSession.sessionObj )
                            this.dbgSession.target.cdic.releaseSession( sessionObj ).submit();
                    }
                }
                
                job.onError = job.onTimeout = function()
                {
                	log("[" + (new Date()).valueOf() + "] DebugSession Error: Cannot acquire session.");
	                this.dbgSession.initializing = false;
	                this.dbgSession.release( true );
	                DebugSession.notifyClients( 'initialized', this.dbgSession.target, this.dbgSession, false );
                }
                
                job.submit();
            }
            catch( exc )
            {
            	log("[" + (new Date()).valueOf() + "] DebugSession Exception: Cannot acquire session." + exc.toString());
                this.initializing = false;
                this.release( true );
                DebugSession.notifyClients( 'initialized', this.target, this, false );
            }
        }
    }
}

DebugSession.prototype.setSessionObj = function( sessionObj )
{
	if( !this.sessionObj || ( this.sessionObj && !this.sessionObj.enabled ) )
	{
		if( sessionObj && !this.releasing )
		{
			this.sessionObj             = sessionObj;
			this.initializing           = false;
			this.sessionObj.onTask      = this.processTasks;
			this.sessionObj.dbgSession  = this;
				
			if( !this.isInitialized )
				DebugSession.notifyClients( 'initialized', this.target, this, true );
		}
	}
}

//-----------------------------------------------------------------------------
// 
// onNotify(...)
// 
// Purpose: Handle broadcast messages
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.onNotify = function( reason )
{
	if( reason == 'initialized' )
		this.isInitialized = true;
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: Release session, stop debugging if active
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.release = function( quiet )
{
    if( this.sessionObj && !this.releasing )
    {
		this.releasing = true;
	
        if( this.isDebugging() )
            this.stop( new Callback( this.doRelease, this ) );
        else
            this.doRelease();
    }
}
    
DebugSession.prototype.doRelease = function( thisObj )
{
    if( !thisObj )
        thisObj = this;
        
	if( thisObj.target && thisObj.target.getFeature( Feature.DISCONNECT, thisObj ) && thisObj.sessionObj )
	{
		var job = thisObj.sessionObj.disconnect();
		job.dbgSession = thisObj;
		job.onResult = job.onError = job.onTimeout = function()
		{
			try
			{
				var job = this.dbgSession.target.cdic.releaseSession( this.dbgSession.sessionObj );
				
				if( job )
				    job.submit();
			}
			catch( exc )
			{}
			
			this.dbgSession.sessionObj = null;
			this.dbgSession.address    = null;
			this.dbgSession.target     = null;
		}
        
		job.submit();
	}
	else
	{
		if( thisObj.target && thisObj.sessionObj )
		{
			var job = thisObj.target.cdic.releaseSession( thisObj.sessionObj );
			
			if( job )
				job.submit();
		}
		
		thisObj.sessionObj = null;
		thisObj.address    = null;
		thisObj.target     = null;
	}
}

//-----------------------------------------------------------------------------
// 
// initialized(...)
// 
// Purpose: Is the session initialized
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.initialized = function()
{
    return ( this.sessionObj != null );
}

//-----------------------------------------------------------------------------
// 
// setDocument(...)
// 
// Purpose: Set current document in debug session
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.setDocument = function( doc, noErrors )
{
	var errorMsg = "";

	if( this.frame == this.stack.length-1 && !noErrors )
	{
		if( this.error )
		{
			errorMsg = this.error;
			log("[" + (new Date()).valueOf() + "] setDocument Error: " + errorMsg);
		}
	}
	
	// Add the doc to my documents list if not present
	var docObj = this.documents[doc.scriptID];
	
	if( !docObj )
		this.documents[doc.scriptID] = docObj = doc;
}

//-----------------------------------------------------------------------------
// 
// setState(...)
// 
// Purpose: Set the state of the debugger with reflection into the UI.
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.setState = function (state)
{
    var oldstate = this.state;		
	this.state = state;

	DebugSession.notifyClients( 'state', this, oldstate, this.state );
}

//-----------------------------------------------------------------------------
// 
// isDebugging(...)
// 
// Purpose: Session in debug mode?
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.isDebugging = function( doc )
{
    var ret = ( this.state != DebugSession.INACTIVE );
    
    if( ret && doc )
        ret = ( this.documents[doc.scriptID] ? true : false );
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// isStopping(...)
// 
// Purpose: Return true if the session is about to stop execution
// 
// Author : pwollek
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.isStopping = function()
{
	return this.stopExe;
}

//-----------------------------------------------------------------------------
// 
// forwardBridgeTalk(...)
// 
// Purpose: Forward a BridgeTalk message to cdi
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.forwardBridgeTalk = function( bt )
{
    if( bt && this.sessionObj)
        this.sessionObj.customCall( 'BridgeTalk', true, bt );
}

//-----------------------------------------------------------------------------
// 
// reset(...)
// 
// Purpose: Reset session
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.reset = function()
{
    this.setState( DebugSession.INACTIVE );

    this.document   = null;
    this.documents  = {};
    this.dbgLevel   = ExecutionContext.DBGLEVEL_BREAK;
    this.line       = -1;
    this.lineColor  = undefined;
    this.frame      = -1;
    this.stack      = '';
    this.error      = null;

	// for UI update
	this.setState( DebugSession.INACTIVE );
}

//-----------------------------------------------------------------------------
// 
// finalizeExecution(...)
// 
// Purpose: Finalize debugging session.
//          Reset the DebugSession, activate the last debugged document
//          and bring app to front
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.finalizeExecution = function()
{    
    if( !this.finalized )
    {
        this.finalized = true;

        var doc = this.document;

        // reset the session
        this.reset();
        
        this.tmpSilentStop = undefined;

        globalBroadcaster.notifyClients( 'shutdown' );
    }

    document = null;
    documents = [];
	
	DebugSession.removeActiveSession();

	if(sessionmanagerglo)
	{
		sessionmanagerglo.invalidate();
	}

    this.finalizing = false;
}

DebugSession.prototype.pumpCDI = function( )
{
	cdi.pump();
	$.sleep (10);
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: Start execution
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.startSession = function( doc, dbgLevel, thisObj, errorInfo )
{
	//
	// store parameters in order to try a restart if the execution failed
	//
	if( thisObj.startInfo )
	{	
		if( thisObj.startInfo.count > DebugSession.kRestartSessionMAX )
			thisObj.startInfo = null;
	}
	else
	{
		thisObj.startInfo = {	
								doc			: doc,
								dbgLevel	: dbgLevel,
								errorInfo	: errorInfo,
								count		: 0
							};
	}
	//
	// if we are waiting for the target to response on
	// a stop command then cancel that now, finalize
	// the previous session and start the new one
	//
	if( thisObj.stopTaskID > -1 )
	{
		thisObj.stopTaskID = -1;
	}

    //
    // thisObj represents "this", but since this function might be also called
    // my a delayed function call we have to pass in the "this"-object
    //
    thisObj.finalized = false;
    
    if( thisObj.sessionObj && thisObj.sessionObj.enabled )
    {   
        //
        // store current doc and debug level
        //
        if( doc )
            thisObj.documents[ doc.scriptID ] = doc;

        thisObj.document       = doc;
        thisObj.dbgLevel       = dbgLevel;
        thisObj.error          = null;
        
        //
        // execution context
        //
        var flags = debugMenu.dontBreakOnErrors == true ? 1 : 0;
        
        var context = null;
		try
		{
			context = new ExecutionContext( doc.scriptID, 
                                            doc.getText(), 
                                            dbgLevel, 
                                            thisObj.profileLevel, 
                                            flags );
		}
		catch( exc )
		{
			context = null;
		}

		if( context )
		{                     
			var breakpoints = [];  
			globalBroadcaster.notifyClients( 'addSessionBreakpoints', thisObj.sessionObj.address, breakpoints );
	        
			for( var i=0; i<breakpoints.length; i++ ){
				context.addBreakpoint( breakpoints[i] );
			}
	        
			//
			// set state of session
			//  
			thisObj.setState( DebugSession.RUNNING );

			//
			// start actual execution
			//
			try
			{
				var job         = thisObj.sessionObj.startExecution( context );
				job.dbgSession  = thisObj;
				job.errorInfo   = errorInfo;

				var keepRunning = true;
	            
				job.onResult = function()
				{	
					if( this.dbgSession.target )
						this.dbgSession.target.prepareDebugSession = false;
					
					if( !this.dbgSession.isStopping() )
					{
						var msg = "Execution finished.";
						var resStr = "";
						var resValue = this.result[0];

						if( !this.result[0] )
							resValue = "undefined";

						resStr = "Result: " + resValue;

						log("[" + (new Date()).valueOf() + "] startSession Info: " + msg + resStr);
						this.dbgSession.stopExe = false;
						this.dbgSession.finalizeExecution();
					}
					keepRunning = false;
				}
	            
				job.onError = function()
				{
					if( this.dbgSession.target )
						this.dbgSession.target.prepareDebugSession = false;
					
					this.dbgSession.stopExe = false;
					
					if( !this.dbgSession.finalized && !this.dbgSession.finalizing )
					{
						var errCode = 0;

						if( !isNaN( this.errorCode ) )
							errCode = this.errorCode;

						var errMsg = this.errorMessage;

						//
						// on error code #1 (engine busy) retry to start the session
						//
						if( errCode == 1 && errMsg == kErrMsgEngineBusy && this.dbgSession.startInfo )
						{
							log("[" + (new Date()).valueOf() + "] startSession Error: " + errMsg);
							this.dbgSession.startInfo.count++;
							this.dbgSession.startSession( this.dbgSession.startInfo.doc, this.dbgSession.startInfo.dbgLevel, this.dbgSession, this.dbgSession.startInfo.errorInfo );
						}
						else
						{
							try
							{
								var errorInfo = this.errorInfo;
								
								if( !errorInfo )
									errorInfo = new ErrorInfo();
							   	
								if( errCode == kErrEngineNotExists ){
									errMsg = localize( "$$$/ESToolkit/Error/EngineNotExists=Engine '%1' does not exists!", thisObj.address.engine );
								}

								if( errCode > 0 )
									errMsg = "(#" + errCode + ") " + errMsg;

								errorInfo.push( errMsg );
								
								var generallMsg = localize( "$$$/ESToolkit/Error/MissingEngine=Cannot execute script in target engine '%1'!", thisObj.address.engine );
								
								if( generallMsg	!= this.errorMessage	&& 
									errCode		!= kErrEngineHalted		&&
									errCode		!= kErrBadAction			)
									errorInfo.push( generallMsg );
							
								this.dbgSession.finalizeExecution();

								if( this.dbgSession.temporary )
								{
									if( errCode == kErrEngineNotExists || errCode == kErrBadAction )
										this.dbgSession.target.removeSession( this.dbgSession );
								}
							}
							catch(exc)
							{}
						}
					}
					keepRunning = false;
				}
	            
				job.submit(-1);	// never timeout
                DebugSession.addActiveSession();
				while(keepRunning == true && (!this.finalized))
				{
					this.pumpCDI();
	        	}

			}
			catch( exc )
			{
				log("[" + (new Date()).valueOf() + "] startSession Exception: " + exc.toString());
				if( this.dbgSession.target )
					thisObj.target.prepareDebugSession = false;
				
				if( thisObj.address )
					errorInfo.push( localize( "$$$/ESToolkit/Error/MissingEngine=Cannot execute script in target engine '%1'!", thisObj.address.engine ) );
	                        
				thisObj.finalizeExecution();
			}
		}
		else if( this.dbgSession.target )
			thisObj.target.prepareDebugSession = false;
    }
    else
    {
        if( this.dbgSession.target )
			thisObj.target.prepareDebugSession = false;

        errorInfo.push( "Can't establish debugging session" );
		
		try
		{
			errorInfo.push( localize( "$$$/ESToolkit/Error/MissingEngine=Cannot execute script in target engine '%1'!", thisObj.address.engine ) );
		}
		catch( exc )
		{}

        thisObj.finalizeExecution();
    }
}

//-----------------------------------------------------------------------------
// 
// DebugSession.command(...)
// 
// Purpose: Continue execution
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.command = function( cmd, dbgContext )
{
    this.setState( DebugSession.RUNNING );
    
    try
    {
        var context = ( dbgContext ? dbgContext : this.createCurrentContext() );
        
        if( cmd != DebugSession.CMD_PAUSE )
        {
        	if(this.error)
        	{	
        		log("[" + (new Date()).valueOf() + "] Error: Runtime error.");
        	}

		    if( this.error )
			   context.flags |= 0x2;
		}
		
		this.error = null;

        var job = null;
        
        switch( cmd )
        {
            case DebugSession.CMD_CONTINUE:
                job = this.sessionObj.continueExecution( context );
                break;
                
            case DebugSession.CMD_PAUSE:
                job = this.sessionObj.pauseExecution( context );
                break;
                
            case DebugSession.CMD_STEPOVER:
                job = this.sessionObj.stepOver( context );
                break;
                
            case DebugSession.CMD_STEPINTO:
                job = this.sessionObj.stepInto( context );
                break;
                
            case DebugSession.CMD_STEPOUT:
                job = this.sessionObj.stepOut( context );
                break;
        }
        
        job.dbgSession  = this;
        job.cmd         = cmd;
		job.dbgContext	= dbgContext;
        
        job.onResult = function()
        {
            // no results are expected
        }
            
        job.onError = function()
        {
            ErrorInfo( localize( "Error during execute command %1", this.cmd ) );
            
            this.dbgSession.stop( undefined, undefined, undefined, this.dbgContext );
        }
        
        job.submit(-1);	// never timeout
    }
    catch( exc )
    {
        var errorInfo = new ErrorInfo( localize( "Error during execute command %1", cmd ) );
        
        this.stop( undefined, undefined, undefined, dbgContext );
    }
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: Stop execution
// 
//-----------------------------------------------------------------------------

DebugSession.postStop = function( dbgSession, callback, silent )
{
	if( dbgSession.isStopping() )
	{
		dbgSession.setState( DebugSession.INACTIVE );

		if( dbgSession && !dbgSession.finalized && !dbgSession.finalizing )
		{
			dbgSession.finalizing = true;
			dbgSession.finalizeExecution();
		}

		dbgSession.stopExe = false;
	    
		if( callback )
			callback.call();
	}
	else if( callback )
		callback.call();
}

DebugSession.prototype.stop = function( callback, silent, force, dbgContext )
{
	//
	// if stop is forced then only silent
	//
	silent = ( force ? force : silent );

	if( !this.isStopping() )
	{
        log("[" + (new Date()).valueOf() + "] Stop execution of the script.");
		this.tmpSilentStop = silent;
		this.stopExe	   = true;

		try
		{
			if( !this.releasing )
			{
				var context    = ( dbgContext ? dbgContext : this.createCurrentContext() );
				var job        = this.sessionObj.stopExecution( context );
				job.dbgSession = this;
				job.callback   = callback;
				job.silent     = silent;
				var keepRunning = true;
                

				if( !force )
				{
					job.onResult = function()
					{
						//
						// if the target application don't respond then
						// finalize the debug session
						//
						if( this.dbgSession.address )
						{
							DebugSession.postStop(this.dbgSession, this.callback, this.silent );
							$.sleep (1000);
						}
						else 
						{
							this.dbgSession.stopExe = false;
							this.dbgSession.setState( DebugSession.INACTIVE );

							if( this.callback )
								this.callback.call();
						}
						keepRunning = false;
					}
			            
					job.onError = job.onTimeout = function()
					{
						if( !this.silent )
						{
							try
							{
								var error = new ErrorInfo( localize( "$$$/ESToolkit/Status/NoRespond=%1 did not respond", this.dbgSession.target.getTitle() ) );
								error.push( kMsgHaltedExecution );
							}
							catch( exc )
							{}
						}
			            
						this.dbgSession.stopExe = false;
						this.dbgSession.setState( DebugSession.INACTIVE );
			            
						this.dbgSession.finalizeExecution();

						if( this.callback )
							this.callback.call();
						keepRunning = false;
					}
				}
				else
					keepRunning = false;

				job.submit( 5000 ); // timeout 5sec.

				while(keepRunning == true && (!this.finalized))
				{
					this.pumpCDI();
				}
			}

			if( force || this.releasing )
			{
				//docMgr.setStatusLine( "" );

				this.stopExe = false;
				this.setState( DebugSession.INACTIVE );
		        
				this.finalizeExecution();

				if( this.callback )
					this.callback.call();
			}
		}
		catch( exc )
		{
			if( !silent )
			{
				var error = new ErrorInfo( localize( "$$$/CT/ExtendScript/Errors/Err59=No response" ) );
				error.push( kMsgHaltedExecution );
			}
	        
			this.stopExe = false;
			this.setState( DebugSession.INACTIVE );
	        
			this.finalizeExecution();

			if( this.callback )
				this.callback.call();
		}
	}
}

//-----------------------------------------------------------------------------
// 
// eval(...)
// 
// Purpose: Eval source
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.eval = function( source, noReply )
{
	if( !this.releasing )
	{
		try
		{
			var job        = this.sessionObj.eval( source );
			job.dbgSession = this;

            job.onResult = function()
			{
				// the reply is datatype,result
                var result = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><result><![CDATA[";
				result += this.result[0];
				result += "]]></result>";
				sessionmanagerglo.setEvalResult(result);
                log("[" + (new Date()).valueOf() + "] eval Result: " + this.result[0]);
			}
	            
			job.onError = function()
			{
				// the reply is the message
				var error = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><error><![CDATA[";
				error += this.errorMessage;
				error += "]]></error>";
				sessionmanagerglo.setEvalResult(error);
				log("[" + (new Date()).valueOf() + "] eval Error: " + this.errorMessage);
			}

			job.onTimeout = function()
			{
			}
	        
			job.submit();
		}
		catch( exc )
		{
			InternalError();
			log("[" + (new Date()).valueOf() + "] eval Exception: " + exc.toString());
		}
	}
}

DebugSession.prototype.sendVariableRequest = function( scope, excludes, max )
{
	var job = null;

	var excludesInternal = 0;
    var excludeList;

    if(excludes != undefined)
    {
        excludeList = excludes.split( ',' );
        for( var i=0; i<excludeList.length; i++ )
        {
            if(excludeList[i] == 'undefined' || excludeList[i] == 'Undefined')
                excludesInternal |= Variable.TYPE_UNDEFINED;
            else if(excludeList[i] == 'builtin' || excludeList[i] == 'Builtin')
                excludesInternal |= Variable.TYPE_CORE;
            else if(excludeList[i] == 'function' || excludeList[i] == 'Function')
                excludesInternal |= Variable.TYPE_FUNCTION;
            else if(excludeList[i] == 'prototype' || excludeList[i] == 'Prototype')
                excludesInternal |= Variable.TYPE_PROTOTYPE;
        }
    }

	try
	{
		job = this.sessionObj.getVariables( scope, excludesInternal, max );
	}
	catch( exc )
	{
		InternalError();
		log("[" + (new Date()).valueOf() + "] sendVariableRequest Exception: " + exc.toString());
	}

	if( job )
	{

		job.onResult = function()
		{
			log("[" + (new Date()).valueOf() + "] sendVariableRequest Successful.");
	   	}
	    
		job.onError = job.onTimeout = function()
		{
			log("[" + (new Date()).valueOf() + "] sendVariableRequest Unsuccessful.");
		}

		job.submit();

	}
	else
	{
		log("[" + (new Date()).valueOf() + "] sendVariableRequest Error: No Job" );
	}
}

DebugSession.prototype.setVariableValue = function( scope, text )
{

	var job	= this.sessionObj.setValue( scope, "dummy" , text );
    
	job.onResult = function()
	{
		log("[" + (new Date()).valueOf() + "] setVariableValue Successful.");
	}
    
	job.onError = job.onTimeout = function()
	{
		log("[" + (new Date()).valueOf() + "] setVariableValue Unsuccessful.");
	}
    
	job.submit();
}



//-----------------------------------------------------------------------------
// 
// resetEngine(...)
// 
// Purpose: Reset the engine
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.resetEngine = function()
{
	if( !this.releasing )
	{
		try
		{
			var job = this.sessionObj.reset();
			job.submit();
		}
		catch( exc )
		{
			InternalError();
		}
	}
}

//-----------------------------------------------------------------------------
// 
// switchFrame(...)
// 
// Purpose: Create execution context for current session
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.switchFrame = function( index )
{
    if( !this.releasing )
    {
        try
        {
            var job = this.sessionObj.switchFrame( index );
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] SwitchFrame Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
//
// getFrame(...)
//
// Purpose: Gets the frame specified by index
//
//-----------------------------------------------------------------------------

DebugSession.prototype.getFrame = function( index )
{
    if( !this.releasing )
    {
        try
        {
            var job = this.sessionObj.getFrame( index );
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] getFrame Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: Create execution context for current session
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.createCurrentContext = function()
{
    var flags = debugMenu.dontBreakOnErrors == true ? 1 : 0;
    var context = null;
	
	if( document )
	{
		context = new ExecutionContext( document.scriptID, 
                                        document.getText(), 
                                        this.dbgLevel, 
                                        this.profileLevel,
                                        flags );
	}

/*  pwollek 04/21/2008: 
    There's no need to collect and send all breakpoints with each debuggign command
    because the breakpoints are sent once when starting the debugging session and
    whenever a breakpoint change happend.
                
    var breakpoints = [];  
    globalBroadcaster.notifyClients( 'addSessionBreakpoints', this.sessionObj.address, breakpoints );
    
    for( var i=0; i<breakpoints.length; i++ )
        context.addBreakpoint( breakpoints[i] );
*/    
    return context;
}

//-----------------------------------------------------------------------------
// 
// setBreakpoints(...)
// 
// Purpose: Set an array of breakpoints
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.setBreakpoints = function( breakpoints )
{
    if( this.sessionObj && !this.releasing )
    {
        try
        {
            var job = this.sessionObj.setBreakpoints( breakpoints );
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] setBreakpoints Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
// 
// getBreakpoints(...)
// 
// Purpose: Get all breakpoints
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.getBreakpoints = function( callback )
{
    if( this.sessionObj && !this.releasing )
    {
        try
        {
            var job = this.sessionObj.getBreakpoints();
            job.cb  = callback;
            
            job.onResult = function()
            {
                if( this.cb )
                    this.cb.call( this.result );
            }
            
            job.onError = job.onTimeout = function()
            {
                if( this.cb )
                    this.cb.call( [] );
            }
            
            job.submit();
        }
        catch( exc )
        {
            if( callback )
                callback.call( [] );
        }
    }
}

//-----------------------------------------------------------------------------
// 
// removeBreakpoints(...)
// 
// Purpose: Remove all breakpoints
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.removeBreakpoints = function()
{
    if( this.sessionObj && !this.releasing )
    {
        try
        {
            var job = this.sessionObj.removeBreakpoints();
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] removeBreakpoints Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
// 
// setBreakpoint(...)
// 
// Purpose: Set a single breakpoint
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.setBreakpoint = function( bp )
{
    if( this.sessionObj && !this.releasing )
    {
        try
        {
            var job = this.sessionObj.setBreakpoint( bp );
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] setBreakpoint Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
// 
// removeBreakpoint(...)
// 
// Purpose: Remove a single breakpoint
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.removeBreakpoint = function( bp )
{
    if( this.sessionObj && !this.releasing )
    {
        try
        {
            var job = this.sessionObj.removeBreakpoint( bp );
            job.submit();
        }
        catch( exc )
        {
            InternalError();
            log("[" + (new Date()).valueOf() + "] removeBreakpoint Exception: " + exc.toString());
        }
    }
}

//-----------------------------------------------------------------------------
// 
// DebugSession(...)
// 
// Purpose: Process incoming tasks
// 
//-----------------------------------------------------------------------------

DebugSession.prototype.processTasks = function( task, dbgSession )
{
    //
    // if this function was called by a delayed task, then 'this' is the 
    // global object and the DebugSession object needs to be passed in.
    // 
    if( !dbgSession )
        dbgSession = this.dbgSession;

	try
	{
		switch( task.name )
		{
			case Job.PRINT:
			{
				print( task.message );
			}
			break;
                    
			case Job.BREAKPOINTS:
			{
			}
			break;
            
			case Job.ERROR:
			{
				dbgSession.error = task.errorMessage;
				print(dbgSession.error);
			}
			break;
            
			case Job.EXECUTION_BREAK:
			case Job.FRAME_SWITCHED:
			{   
				//BTBackend directly send the break xml to client
			}
			break;
            
			case Job.EXECUTION_EXIT:
			{
				dbgSession.stopExe = false;

				if( !dbgSession.finalized && !dbgSession.finalizing )
				{
					log("[" + (new Date()).valueOf() + "] Execution Finished.");
					dbgSession.finalizeExecution();
				}

				task.quitTask();
			}
			break;
		}
	}
	catch( exc )
	{}
}
