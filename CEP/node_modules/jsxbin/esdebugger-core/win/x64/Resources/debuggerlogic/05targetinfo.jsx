/**************************************************************************
*
*  @@@BUILDINFO@@@ 05targetinfo-2.jsx 3.5.0.7		08-December-2008
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

///////////////////////////////////////////////////////////////////////////////
//
// class TargetManager
//

//-----------------------------------------------------------------------------
// 
// TargetManager(...)
// 
// Purpose: ctor
// 
//-----------------------------------------------------------------------------

function TargetManager( defaultTarget, defaultTitle )
{
    this.targetsInfo    = new Array;
    this.defaultTarget  = null;
    this.activeTarget   = null;

    this.broadcaster    = new Broadcaster;    
    
    this.changing       = 0;
	
	globalBroadcaster.registerClient( this, 'shutdown' );

    this.defaultTarget  = this.getActiveTarget();
    this.activeTarget   = this.defaultTarget;
}

//-----------------------------------------------------------------------------
// 
// onNotify(...)
// 
// Purpose: Process broadcast messages
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.onNotify = function( reason )
{
	if( reason == 'shutdown' )
	{
	    globalBroadcaster.unregisterClient( this );
		this.unregisterAllClients();
	}
}

//-----------------------------------------------------------------------------
// 
// queryExit(...)
// 
// Purpose: Check for any open debug session. If found, ask the user if 
//          ask is true. If it is OK to stop debugging, and, if so, 
//          terminate debugging.
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.queryExit = function( ask )
{
	for( var i=0; i<this.targetsInfo.length; i++ )
	{
		if( this.targetsInfo[i].isDebugging() )
		{
			if( !ask )
				return false;
			else
				break;
		}
	}

	for( i=0; i<this.targetsInfo.length; i++ )
	{
	    for( var j=0; j<this.targetsInfo[i].sessions.length; j++ )
	    {
	        if( this.targetsInfo[i].sessions[j].isDebugging() )
	            this.targetsInfo[i].sessions[j].stop( undefined, true );
	    }
	}

	return true;
}

//-----------------------------------------------------------------------------
// 
// addTarget(...)
// 
// Purpose: Add a target
// 
//-----------------------------------------------------------------------------

var exAppVerCS3 = { 'aftereffects'  : '8.0',
                    'soundbooth'    : '1.0',
                    'acrobat'       : '8.0',
                    'contribute'    : '4.5',
                    'dreamweaver'   : '9.0',
                    'fireworks'     : '9.0',
                    'flash'         : '9.0',
                    'bridge'        : '2.0',
                    'devicecentral' : '1.0',
                    'encore'        : '3.0',
                    'estoolkit'     : '2.0',
                    'illustrator'   : '13.0',
                    'indesign'      : '5.0',
                    'indesignserver': '5.0',
                    'incopy'        : '5.0',
                    'photoshop'     : '10.0',
                    'stockphotos'   : '1.5',
                    'audition'      : '3.0',
                    'premiere'      : '3.0'
                  };

var exAppVerCS4 = { 'aftereffects'		: '9.0',
					'soundbooth'		: '2.0',
					'acrobat'			: '9.0',
					'contribute'		: '5.0',
					'dreamweaver'		: '10.0',
					'fireworks'			: '10.0',
					'flash'				: '10.0',
					'bridge'			: '3.0',
					'devicecentral'		: '2.0',
					'encore'			: '4.0',
					'estoolkit'			: '3.0',
					'illustrator'		: '14.0',
					'indesign'			: '6.0',
					'indesignserver'	: '6.0',
					'incopy'			: '6.0',
					'photoshop'			: '11.032',
					'photoshop'			: '11.064',
					'stockphotos'		: '2.0',
					'premiere'			: '4.0',
					'exman'				: '2.0',
					'ame'				: '4.0'
				  };

TargetManager.prototype.addTarget = function( address, checkConnectState, dynamic, quiet )
{
    var info = this.findTarget( address );
    
    if( !info )
    {
        //
        // check for excluded application versions
        //
        var excluded = false;
		var exCS3	 = false;     //!prefs.cmp.CS3.getValue( Preference.BOOLEAN );
		var exCS4	 = false      //!prefs.cmp.CS4.getValue( Preference.BOOLEAN );

		// if( address.script == "cs/3" && exCS3 )
		// {
		//     var target  = address.target.split ('-');
		//     var version = target [1];
		//     var locale  = target [2];
		//     target      = target [0];

  //           excluded |= ( exAppVerCS3[target] == version );
		// }

		// if( address.script == "cs/4" && exCS4 )
		// {
		//     var target  = address.target.split ('-');
		//     var version = target [1];
		//     var locale  = target [2];
		//     target      = target [0];

  //           excluded |= ( exAppVerCS4[target] == version );
		// }

        if( !excluded )
        {
            info = new TargetInfo( address, this, checkConnectState, dynamic );
            this.targetsInfo.push( info );

            if( !quiet )	    
                this.notifyClients( 'addTarget', info );
        }
    }
	else
	{
		info.addSession( address );
	}
    
    if( !this.defaultTarget && info && info.address.target == 'estoolkit-4.0' )
        this.defaultTarget = info;
        
    return info;
}

//-----------------------------------------------------------------------------
// 
// removeTarget(...)
// 
// Purpose: Remove dynamic target
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.removeTarget = function( target )
{
    //
    // only remove targets with dynamic flag set
    //
    if( target && target.dynamic )
    {
        //
        // first remove all sessions
        //
        while( target.sessions.length > 0 )
            target.removeSession( target.sessions[0] );
            
        for( var i=0; i<this.targetsInfo.length; i++ )
        {
            if( this.targetsInfo[i] == target )
            {
                this.targetsInfo.splice(i,1);

                if( this.activeTarget == target )
                    this.activeTarget = this.defaultTarget;
                    
                this.notifyClients( 'changeTargets', this );
                break;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// 
// findTarget(...)
// 
// Purpose: Find a target by address
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.findTarget = function( address )
{
    for( var i=0; i<this.targetsInfo.length; i++ )
    {
        if( this.targetsInfo[i].address.type     == address.type     &&
            this.targetsInfo[i].address.target   == address.target   &&
            this.targetsInfo[i].address.instance == address.instance     )
            return this.targetsInfo[i];
    }
           
    return null;
}

//-----------------------------------------------------------------------------
// 
// findTargetBySpecifier(...)
// 
// Purpose: Find a target by BridgeTalk specifier
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.findTargetBySpecifier = function( specifier )
{
    var target = null;
    
    for( var i=0; i<this.targetsInfo.length; i++ )
    {
        if( this.targetsInfo[i].address.target == specifier )
        {
            target = this.targetsInfo[i];
            break;
        }
    }
    
    return target;
}

//-----------------------------------------------------------------------------
// 
// findSessionSpecifier(...)
// 
// Purpose: Find session by its targets BridgeTalk specifier and engine name
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.findSessionSpecifier = function( specifier, engine )
{
    var sessions = [];
    
    var target = this.findTargetBySpecifier( specifier );
    
    if( target )
    {
        var session = target.findSessionEngine( engine );
        
        if( session )
            sessions.push( session );
    }
    
    return sessions;
}

//-----------------------------------------------------------------------------
// 
// getTargetDisplayname(...)
// 
// Purpose: Return display name of passed taregt
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.getTargetDisplayName = function( target )
{
    if( target && typeof target == 'string' )
        target = this.findTargetBySpecifier( target );
        
    var ret = '';
    
    if( target )
        ret = target.getTitle();
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// setActive(...)
// 
// Purpose: Set active target & session
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.setActive = function( target, session, quiet, dontConnect )
{
    if( !session )
        session = null;
        
    if( target )
    {
        var old = this.activeTarget;
        this.activeTarget = target;

        if( !quiet )
            this.notifyClients( 'changeActiveTarget' );
           
        target.setActive( session, quiet, dontConnect );
    }
}

//-----------------------------------------------------------------------------
// 
// getActiveSession(...)
// 
// Purpose: Return current active session
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.getActiveSession = function()
{
    var ret    = null;
    var target = this.getActiveTarget();
    
    if( target )
        ret = target.getActive();
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// getActiveTarget(...)
// 
// Purpose: Return current active target
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.getActiveTarget = function()
{
    if( !this.activeTarget )
    {    
        if( !this.defaultTarget )
            this.addTarget( new Address( 'BTBackend', 'estoolkit-4.0', '', '', 'main', 'ExtendScript Toolkit CC' ), true, false, true );

        this.setActive( this.defaultTarget );
    }
        
    return this.activeTarget;
}

//-----------------------------------------------------------------------------
// 
// getTargets(...)
// 
// Purpose: return array of targets
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.getTargets = function()
{
    var targets = new Array;
    
    for( var i=0; i<this.targetsInfo.length; i++ )
        targets.push( this.targetsInfo[i] );
                
    return targets;
}

//-----------------------------------------------------------------------------
//
// getTargetsAsXML(...)
//
// Purpose: return array of targets as XML
//
//-----------------------------------------------------------------------------

TargetManager.prototype.getTargetsAsXML = function()
{
    var xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><targets>";

    for( var i=0; i<this.targetsInfo.length; i++ ) {
        xml += "<target specifier=\"";
        xml += this.targetsInfo[i].address.target;
        xml += "\">";
        xml += this.targetsInfo[i].address.label;
        xml += "</target>";
    }

    xml += "</targets>"

    return xml;
}

//-----------------------------------------------------------------------------
//
// getTargetAppRunning(...)
//
// Purpose: return if the target specifier app is runnnig
//
//-----------------------------------------------------------------------------
TargetManager.prototype.getTargetAppRunning = function(targetSpecifier)
{
    var target = this.findTargetBySpecifier(targetSpecifier);
    if(target)
    {
        return target.isRunning();
    }
    return false;
}

//-----------------------------------------------------------------------------
// 
// setConnected(...)
// 
// Purpose: get/set connect status of target
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.setConnected = function( addr, connected, quiet )
{
    if( !connected )
        connected = false;
        
    var target = this.findTarget( addr );
    
    if( target )
        target.setConnected( connected, quiet );
}

TargetManager.prototype.getConnected = function( addr, check )
{
    var ret = false;
    
    var target = this.findTarget( addr );
    
    if( target )
        ret = target.getConnected( check );
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// checkConnections(...)
// 
// Purpose: Check if the connected targets are still running, otherwise 
//          disconnect debugger for target
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.checkConnections = function()
{
    for( var i=0; i<this.targetsInfo.length; i++ )
	{
		//
		// don't check for the ESTK itself
		//
		if( this.defaultTarget != this.targetsInfo[i] )
	        this.targetsInfo[i].checkConnection();
	}
}

//-----------------------------------------------------------------------------
// 
// loadTargets(...)
// 
// Purpose: load all available targets
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.loadTargets = function()
{
    var cdics = cdicMgr.getCDICs();
    
    this.tmpTargetsToLoad = 0;
    
    for( var i=0; i<cdics.length; i++ )
        if( cdics[i].features[Feature.TARGET_ADDRESSES].supported )
            this.tmpTargetsToLoad++;

    for( var i=0; i<cdics.length; i++ )
    {
        if( cdics[i].features[Feature.TARGET_ADDRESSES].supported )
        {
            try
            {
                var job       = cdics[i].getTargetAddresses();
                job.targetMgr = this;
                job.onResult = function()
                {
                    if( this.result )
                    {
                        for( var j=0; j<this.result.length; j++ )
                            this.targetMgr.addTarget( this.result[j], false, false, true );
                            
                        if( --this.targetMgr.tmpTargetsToLoad <= 0 )
                        {
                            if( this.targetMgr.targetsInfo.length > 0 )
                                this.targetMgr.setActive( this.targetMgr.defaultTarget );
                            
                            this.targetMgr.notifyClients( 'changeTargets' );
                        }
                    }
                }
                
                job.submit(true);
            }
            catch( exc )
            {
                log("[" + (new Date()).valueOf() + "] loadTargets Exception: " + exc.toString());
            }
        }
    }
}

//-----------------------------------------------------------------------------
// 
// sendBreakpoints(...)
// 
// Purpose: Send all breakpoints to target/session
// 
//-----------------------------------------------------------------------------


TargetManager.prototype.sendBreakpoints = function( target, session )
{
    if( !target )
        target = this.getActiveTarget();

    if( target )
    {
        if( !session || !target.exists( session ) )
            session = target.getActive();

        if( session )
        {
            var breakpoints = [];  
            globalBroadcaster.notifyClients( 'addSessionBreakpoints', session.address, breakpoints );
            session.setBreakpoints( breakpoints );
        }
    }
}

//-----------------------------------------------------------------------------
// 
// registerClient(...)
// 
// Purpose: un/register client object to receive notifications
//                   (client objects have to support a function "onTargetsChanged(reason)" )
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.registerClient = function( clientObj, msg )
{ return this.broadcaster.registerClient( clientObj, msg ); }

TargetManager.prototype.unregisterClient = function( clientObj, msg )
{ this.broadcaster.unregisterClient( clientObj, msg ); }

TargetManager.prototype.unregisterAllClients = function()
{ this.broadcaster.unregisterAllClients(); }

//-----------------------------------------------------------------------------
// 
// notifyClients(...)
// 
// Purpose: notify registered client object about changes related to the
//                   targets list
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.notifyClients = function( why, currentTarget )
{
    if( !currentTarget )
        currentTarget = this.getActiveTarget();
		
	var toFront = true;
	
	if( why == 'endConnect' )
	{
		if( currentTarget.prepareDebugSession )
			toFront = false;
	}

    this.broadcaster.notifyClients( why, currentTarget );        
    
    //
    // default actions
    //
    switch( why )
    {       
        case 'targetDied':
            this.removeTarget( currentTarget );
            break;
    }
}

//-----------------------------------------------------------------------------
// 
// resetChangingFlag(...)
// 
// Purpose:changing flag
// 
//-----------------------------------------------------------------------------

TargetManager.prototype.resetChangingFlag = function()
{
    this.changing = 0;
}

TargetManager.prototype.setIsChanging = function( changing )
{
    if( changing )
        this.changing++;
    else
        this.changing--;
}

TargetManager.prototype.isChanging = function()
{
    return this.changing > 0;
}

///////////////////////////////////////////////////////////////////////////////
//
// class TargetInfo
//

//-----------------------------------------------------------------------------
// 
// TargetInfo(...)
// 
// Purpose: ctor
// 
//-----------------------------------------------------------------------------

function TargetInfo( address, mgr, updateConnectState, dynamic )
{
    this.mgr                = mgr;      // target manager
    this.sessions           = [];       // sessions of target
    this.activeSession      = null;     // current active session
    this.connected          = false;    // connection state of target
    this.changeConnectState = false     // true if about to change the current connection state
    this.dynamic            = dynamic;  // dynamic targets gets removed from the list
                                        // on receiving the command REMOVE_ADDRESS (see 04cdicmanager.jsx)
                                        // or if it's last session was removed
										
	this.prepareDebugSession= false;	// true if the target is about to connect for a debug session
    
    this.address            = address;
    this.cdic               = cdicMgr.cdic[ this.address.type ];
    this.fileType           = [];       // executeable file types (file name extensions)
    
    this.isESTarget         = ( address.type == 'BTBackend' );
    
    if( this.address.engine.length > 0 )
        this.addSession( this.address, null, true );

    if( !this.cdic.fileTypes )
    {
        if( this.getFeature( Feature.GET_FILE_TYPES ) )
        {        
            try
            {
                //
                // ask for executable file types
                //
                var job    = this.cdic.getFileTypes();
                job.target = this;
                
                job.onResult = function()
                {
                    this.target.cdic.fileTypes = this.result;
                    
                    for( var i=0; i<this.result.length; i++ )
                        this.target.addFiletype( this.result[i] );
                }
                
                job.submit();
            }
            catch( exc )
            {
                log("[" + (new Date()).valueOf() + "] TargetInfo:getFileTypes Exception: " + exc.toString());
            }
        }
    }
    else
    {
        for( var i=0; i<this.cdic.fileTypes.length; i++ )
            this.addFiletype( this.cdic.fileTypes[i] );
    }
        
    if( updateConnectState && !this.getChangeConnectState() && this.cdic && this.sessions.length )
    {
        this.setChangeConnectState( true );
        
        try
        {
            //
            // first check if target app is running
            //
            var job         = this.cdic.isTargetRunning( this.address );
            job.target      = this;
            
            job.onResult = function()
            {
                this.target.connected = this.result[0];
                this.target.setChangeConnectState( false );
            }
                
            job.onError = job.onTimeout = function()
            {
                this.target.connected = false;
                this.target.setChangeConnectState( false );
                log("[" + (new Date()).valueOf() + "] TargetInfo:isTargetRunning Error.");
            }
            
            job.submit();
        }
        catch( exc )
        {
            this.connected = false;
            this.setChangeConnectState( false );
            log("[" + (new Date()).valueOf() + "] TargetInfo Exception: " + exc.toString());
        }
    }
}

TargetInfo.prototype.getTitle = function()
{
    var ret = '';
    
    if( this.address )
    {
        ret = this.address.label;
        
        if( ret.length <= 0 )
            ret = this.address.target.substr(0,1).toUpperCase() + this.address.target.substr(1);
    }
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// isDefault(...)
// 
// Purpose: Am I the default target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.isDefault = function()
{
    var ret = false;
    
    if( this.mgr )
        ret = ( this.mgr.defaultTarget == this );
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// removeAllSessions(...)
// 
// Purpose: remove all existing sessions
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.removeAllSessions = function( quiet )
{
    //
    // remove all sessions
    //
    while( this.sessions.length > 0 )
        this.removeSession( this.sessions[0], true );
    
    this.activeSession = null;
    
    //
    // notify clients
    //
    if( !quiet )
        this.mgr.notifyClients( 'removeSessions', this );
}

//-----------------------------------------------------------------------------
// 
// addSessions(...)
// 
// Purpose: Add sessions to target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.addSessions = function( sessionAddrArray, clear, quiet )
{
    var old = this.activeSession;

    //
    // remove all existing sessions
    //    
    if( clear )
        this.removeAllSessions();
   
    //
    // create and add new sessions
    //
    for( var i=0; i<sessionAddrArray.length; i++ )
	{
        this.addSession( sessionAddrArray[i], null, true );
    }
	
    //
    // set active session
    //
	if( old && this.exists( old ) )
	    this.activeSession = old;
	else if( this.sessions.length > 0 )
	    this.activeSession = this.sessions[0];

    if( !quiet )
        this.mgr.notifyClients( 'changeSessions', this );	    
}

//-----------------------------------------------------------------------------
// 
// addSession(...)
// 
// Purpose: Add single session to target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.addSession = function( sessionAddr, sessionObj, quiet )
{
	// no engine, no fun!
	if( sessionAddr && sessionAddr.engine.length <= 0 )
		return null;
		
    var session = this.findSession( sessionAddr );
    
    if( !session )
    {        
        session = new DebugSession( sessionAddr, this, sessionObj );
        this.sessions.push( session );
        
        if( !this.activeSession )
        {
            this.activeSession = session;

        }            

        if( !quiet )
            this.mgr.notifyClients( 'changeSessions', this );	    
    }
    else if( sessionObj )
    {
        session.setSessionObj( sessionObj );

        if( !quiet )
            this.mgr.notifyClients( 'changeSessions', this );	    
    }

	return session;
}

//-----------------------------------------------------------------------------
// 
// removeSession(...)
// 
// Purpose: Remove single session from target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.removeSession = function( session, quiet )
{
    for( var i=0; i<this.sessions.length; i++ )
    {
        if( this.sessions[i] == session )
        {
            session.release();
            this.sessions.splice(i,1);
            
            if( this.activeSession == session )
            {
                if( this.sessions.length > 0 )
                    this.activeSession = this.sessions[0];
                else
                    this.activeSession = null;
            }
        
            if( !quiet )    
                this.mgr.notifyClients( 'changeSessions', this );
                
            break;
        }
    }
}

//-----------------------------------------------------------------------------
// 
// changeSession(...)
// 
// Purpose: Change address of session
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.changeSession = function( oldAddress, newAddress )
{
    for( var i=0; i<this.sessions.length; i++ )
    {
        if( this.sessions[i].address.type     == session.address.type     &&
            this.sessions[i].address.target   == session.address.target   &&
            this.sessions[i].address.instance == session.address.instance &&
            this.sessions[i].address.engine   == session.address.engine     )
        {
            session.address = newAddress;

            this.mgr.notifyClients( 'changeSessions', this );
            break;
        }
    }
}

//-----------------------------------------------------------------------------
// 
// find(...)
// 
// Purpose: Find array index of given session
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.find = function( session )
{
    var ret = -1;
    
    if( session )
    {
        for( var i=0; i<this.sessions.length; i++ )
        {
            if( this.sessions[i] == session                                     ||
                ( this.sessions[i].address.type     == session.address.type     &&
                  this.sessions[i].address.target   == session.address.target   &&
                  this.sessions[i].address.instance == session.address.instance &&
                  this.sessions[i].address.engine   == session.address.engine   )   )
            {
                ret = i;
                break;
            }
        }
    }
    
    return ret;
}

//-----------------------------------------------------------------------------
// 
// findSession(...)
// 
// Purpose: Find session by address
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.findSession = function( sessionAddr )
{
    var ret = null;
    
    if( sessionAddr )
    {
        for( var i=0; i<this.sessions.length; i++ )
        {
            if( this.sessions[i].address.type       == sessionAddr.type     &&
                this.sessions[i].address.target     == sessionAddr.target   &&
                this.sessions[i].address.instance   == sessionAddr.instance &&
                this.sessions[i].address.engine     == sessionAddr.engine     )
            {
                ret = this.sessions[i];
                break;
            }
        }
    }
    
    return ret;
}

//-----------------------------------------------------------------------------
// 
// findSessionEngine(...)
// 
// Purpose: Find session by its engine name
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.findSessionEngine = function( engine )
{
    var ret = null;
    
    for( var i=0; i<this.sessions.length; i++ )
    {
        if( this.sessions[i].address.engine == engine )
        {
            ret = this.sessions[i];
            break;
        }
    }

    if(ret == null) 
    {
        // New engine name. Create session with this engine.
        var newAddr     = new Address( this.address );
        newAddr.engine  = engine;
        targetMgr.addTarget( newAddr, false, false );
        for( var i=0; i<this.sessions.length; i++ )
        {
            if( this.sessions[i].address.engine == engine )
            {
                ret = this.sessions[i];
                break;
            }
        }
    }
    
    return ret;
}

//-----------------------------------------------------------------------------
// 
// exists(...)
// 
// Purpose: given session already in target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.exists = function( session )
{
    return ( this.find( session ) >= 0 );
}

//-----------------------------------------------------------------------------
// 
// setActive(...)
// 
// Purpose: set active session
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.setActive = function( session, quiet, dontConnect )
{
    //
    // if no session passed to set as the active one
    // then take the first session from the list
    //
    if( !session )
    {
        if( this.sessions.length > 0 )
            session = this.sessions[0];
    }

    //
    // check if session is related to this target before
    // set as the active one
    //
    if( this.exists( session ) )
    {
        this.activeSession = session;
    }        
    else
        this.activeSession = null;

    //
    // broadcast change
    //
    if( !quiet )	    
        this.mgr.notifyClients( 'changeActiveSession', this );

    if(!this.isRunning())
    {
        this.setConnected(false);
    }
    //
    // if target is running but not connected yet, then try
    // to connect
    //  
    if( !this.getConnected() && !dontConnect )
	{
		var err = new ErrorInfo();
		var cb = new Callback( function(err){ }, err );

        this.connect( false, cb, err );
	}
}

//-----------------------------------------------------------------------------
// 
// getActive(...)
// 
// Purpose: get active session
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.getActive = function()
{
    if( !this.activeSession )
        return null;
        
    return this.activeSession;
}

//-----------------------------------------------------------------------------
// 
// getSessions(...)
// 
// Purpose: return session list
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.getSessions = function()
{
    return this.sessions;
}

//-----------------------------------------------------------------------------
//
// get all session engines list as xml(...)
//
// Purpose: return all sessions engines
//
//-----------------------------------------------------------------------------

TargetInfo.prototype.getSessionsEnginesAsXML = function()
{
    var xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><engines";
    var targetSpecifier = "";
    if(!this.sessions || this.sessions.length <= 0)
    {
        xml += "></engines>";
        return xml;
    }

    targetSpecifier = this.sessions[0].address.target;
    xml += " targetSpecifier=\"" + targetSpecifier + "\">";

    for( var i=0; i<this.sessions.length; i++ )
    {
        var engine = this.sessions[i].address.engine;
        xml += "<engine>" + engine + "</engine>";
    }

    xml += "</engines>"
    return xml;
}

//-----------------------------------------------------------------------------
//
// isRunning(...)
//
// Purpose: return if the target is running or not[synchronously]
//
//-----------------------------------------------------------------------------

TargetInfo.prototype.isRunning = function()
{
    var running = false;
    try
    {

        var job         = this.cdic.isTargetRunning( this.address );
        job.target      = this;
        var abortTargetRunningJob = false;

        job.onResult = function()
        {
            if( this.result[0] )
            {
                running = true;
            }
            abortTargetRunningJob = true;
        }

        job.onError = job.onTimeout = function()
        {
            abortTargetRunningJob = true;
            log("[" + (new Date()).valueOf() + "] isRunning:isTargetRunning Error.");
        }

        job.submit();
        while( !abortTargetRunningJob )
        {
            cdi.pump();
            $.sleep (10);
        }
    }
    catch( exc )
    {
        log("[" + (new Date()).valueOf() + "] isRunning Exception: " + exc.toString());
    }
    return running;
}

//-----------------------------------------------------------------------------
// 
// connect(...)
// 
// Purpose: connect/disconnect target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.connect = function( dontLaunch, callback, errorInfo, force, ignoreCmp )
{
	if( this.mgr.defaultTarget == this )
	{
	    if( callback )
	        callback.call( true );
	        
		return;
	}
		
    if( !dontLaunch )	dontLaunch	= false;
	if( !ignoreCmp )	ignoreCmp	= false;

	//
	// If the target is a CS3/CS4 target then it is not allowed to have an ESTK CS3/CS4 running.
	// That is to prevent that ESTK CS3/CS4 setup a connection to the target before.
	//
	if( !ignoreCmp && this.address.script )
	{
		var estkSpec = "";

		switch( this.address.script )
		{
			case "cs/3":	estkSpec = "estoolkit-2.0";	break;
			case "cs/4":	estkSpec = "estoolkit-3.0";	break;
		}

		if( estkSpec && BridgeTalk.isRunning( estkSpec ) )
		{
			if( errorInfo )
			{
				var label = BridgeTalk.getDisplayName( estkSpec );

				if( !label )
					label = estkSpec;

				errorInfo.push( localize( "$$$/ESToolkit/Alerts/WhyCannotConnect1=In order to establish a connection between target %1 and the %2 you have to shutdown %3 first!", this.getTitle(), targetMgr.defaultTarget.getTitle(), label ) );
				errorInfo.push( localize( "$$$/ESToolkit/Alerts/CannotConnect=Cannot connect to target %1!", this.getTitle() ) );
			}

			if( callback )
				callback.call( false );

			return;
		}
	}

    if( ( !this.getConnected() || ( this.getConnected() && this.sessions.length == 0 ) ) && 
		!this.getChangeConnectState() && this.cdic )
    {
        this.setChangeConnectState( true );
        
        try
        {
            //
            // first check if target app is running
            //
            var job         = this.cdic.isTargetRunning( this.address );
            job.target      = this;
            job.dontLaunch  = dontLaunch;
            job.cb          = callback;
			job.errorInfo	= errorInfo;
			job.force		= force;
			job.ignoreCmp	= ignoreCmp;
            var abortTargetRunningJob = false;
            
            job.onResult = function()
            {
                if( this.result[0] == true )
                {
					if( !this.force && !this.ignoreCmp && 
						( this.target.address.script == "cs/3" || this.target.address.script == "cs/4" ) )
					{
						//
						// if the target is a CS3/CS4 target then it is required that the target application
						// gets launched by the ESTK CS5!
						// That is to be sure that ESTK CS5 connects to the target first and no other ESTK (CS3/CS4)
						// connected to the target earlier.
						//
						this.target.setChangeConnectState( false );

						if( errorInfo )
						{
							errorInfo.push( localize( "$$$/ESToolkit/Alerts/ShutdownConnect=Please shutdown %1 and try to connect again.", this.target.getTitle() ) );
							errorInfo.push( localize( "$$$/ESToolkit/Alerts/WhyCannotConnect2=In order to establish a connection between target %1 and the %2 the target application needs to be launched by the %2.", this.target.getTitle(), targetMgr.defaultTarget.getTitle() ) );
                            errorInfo.push( localize( "$$$/ESToolkit/Alerts/CannotConnect=Cannot connect to target %1!", this.target.getTitle() ) );
						}

						if( callback )
							callback.call( false );
					}
					else
					{
						//
						// target app is running, now connect
						//
						this.target.doConnect( [ this.target, this.cb ], true, this.errorInfo );
					}
                }
                else
                {
                    if( !this.dontLaunch )
                    {
                        //
                        // first launch target, then connect
                        //
                        appInstance.launchTargetAppSynchronous( this.target, true, new Callback( this.target.doConnect, [ this.target, this.cb ] ) );
                    }
                    else
					{
						if( callback )
							callback.call( false );

                        this.target.setChangeConnectState( false );
					}
                }
                abortTargetRunningJob = true;
            }
                
            job.onError = job.onTimeout = function()
            {
                if( errorInfo ){
                    errorInfo.push( this.errorMessage );
                }
                if( callback )
                    callback.call( false );

                this.target.setChangeConnectState( false );
                abortTargetRunningJob = true;
            }
            
            job.submit();
            while( !abortTargetRunningJob )
            {
                cdi.pump();
                $.sleep (20);
            }
        }
        catch( exc )
        {
            if( errorInfo ){
                errorInfo.push( localize( "$$$/CT/ExtendScript/Errors/Err33=Internal error" ) );
            }
            if( callback )
                callback.call( false );
                
            this.setChangeConnectState( false );
        }
    }
    else if( callback )
	    callback.call( true );
}

TargetInfo.prototype.doConnect = function( args, running )
{
    var thisTarget  = args[0];
    var callback    = args[1];
    var errorInfo   = ( callback && callback.args && callback.args.error ? callback.args.error : new ErrorInfo() );
    
    thisTarget.mgr.notifyClients( 'startConnect', this );
    if( running && thisTarget.cdic )
    {
        try
        {
            //
            // initialize target
            //
            var job         = thisTarget.cdic.initializeTarget( args[0].address );
            job.target      = thisTarget;
            job.cb          = args[1];
            job.errorInfo   = errorInfo;

            var outerAbort = false;
            
            job.onResult = function()
            {
                var init = this.result[0];

                if( init )
                {
                    try
                    {
                        //
                        // when target is initialized ask for session addresses
                        //
                        var job         = this.target.cdic.getSessionAddresses( this.target.address );
                        job.target      = this.target;
                        job.cb          = this.cb;
                        job.errorInfo   = this.errorInfo;
                        var innerAbort  = false;
                        
                        job.onResult = function()
                        {       
                            //
                            // for each received session address create
                            // a session object
                            //
                            this.target.addSessions( this.result, false );
                            
                            if( this.cb )
                                this.cb.call();
                                
                            this.target.setChangeConnectState( false );
                            this.target.mgr.notifyClients( 'endConnect', this.target );
                            innerAbort  = true;
                        }
                        
                        job.onError = job.onTimeout = function()
                        {
                            this.target.setChangeConnectState( false );
                            this.target.mgr.notifyClients( 'endConnect', this.target );
                            
                            if( this.cb )
                            {
                                this.errorInfo.push( this.errorMessage );
                                this.cb.call();
                            }
                            innerAbort = true;
                        }

                        job.submit();
                        while( !innerAbort )
                        {
                            cdi.pump();
                            $.sleep (20);
                        }
                    }
                    catch( exc )
                    {
                        this.target.setChangeConnectState( false );
                        
                        if( this.cb )
                        {
                            this.errorInfo.push( localize( "$$$/CT/ExtendScript/Errors/InternalError=InternalError" ) );
                            this.cb.call();
                        }
                    }
                }
                
                //
                // set connection state
                //
                this.target.setConnected( init );
                outerAbort = true;
            }
                
            job.onError = job.onTimeout = function()
            {
                this.target.setChangeConnectState( false );
                this.target.mgr.notifyClients( 'endConnect', this.target );

                if( this.cb )
                {
                    this.errorInfo.push( this.errorMessage );
                    this.cb.call();
                }
                outerAbort = true;
            }

            job.submit();
            while( !outerAbort )
            {
                cdi.pump();
                $.sleep (10);
            }
        }
        catch( exc )
        {
            thisTarget.setChangeConnectState( false );
            thisTarget.mgr.notifyClients( 'endConnect', thisTarget );

            if( callback )
            {
                errorInfo.push( localize( "$$$/CT/ExtendScript/Errors/InternalError=InternalError" ) );
                callback.call();
            }
        }
    }
    else
    {
        thisTarget.setChangeConnectState( false );
        thisTarget.mgr.notifyClients( 'endConnect', thisTarget );

        if( callback )
        {
            callback.call();
        }
    }
}

TargetInfo.prototype.disconnect = function( force, quiet )
{
	if( this.mgr.defaultTarget == this )
		return;

    if( this.getConnected() || this.dynamic )
    {
        //
        // is any of the targets sessions in debug mode
        //
        var isDebugging = this.isDebugging();
        
        if( force && isDebugging )
        {
            for( var i=0; i<this.sessions.length; i++ )
            {
                if( this.sessions[i].isDebugging() )
                    this.sessions[i].stop( undefined, true, true );
            }
        }
        
        this.setChangeConnectState( true );
        
        //
        // release sessions
        //
        for( var i=0; i<this.sessions.length; i++ )
            this.sessions[i].release( quiet );
          
        //
        // remove sessions from target
        //  
        this.removeAllSessions( quiet );
          
        //
        // set connection state
        //  
        this.setConnected( false, quiet );
        
        //
        // tell CDIC
        //
        if( this.cdic )
            this.cdic.exitTarget( this.address ).submit();
			
	    this.setChangeConnectState( false );
          
        //
        // notify disconnection state
        //  
        if( !quiet )	    
		    this.mgr.notifyClients( 'targetDied', this );
    }
}

//-----------------------------------------------------------------------------
// 
// setConnected(...)
// 
// Purpose: get/set connect status of target
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.setConnected = function( connected, quiet )
{
    if( !connected )
        connected = false;
        
    this.connected = connected;
    
    if( !quiet )
        this.mgr.notifyClients( 'changeConnectionState', this );
}

TargetInfo.prototype.getConnected = function()
{
    return this.connected;
}

//-----------------------------------------------------------------------------
// 
// checkConnection(...)
// 
// Purpose: Check connection to target. If target isn't answering the disconnect
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.checkConnection = function( force )
{
	if( !this.checkingConnection )
	{
		if( !force )
			force = false;

		if( ( this.connected || this.dynamic ) && !this.getChangeConnectState() && this.cdic )
		{
			this.checkingConnection = true;

			try
			{
				//
				// first check if target app is running
				//
				var job         = this.cdic.isTargetRunning( this.address );
				job.target      = this;
				job.force		= force;
				job.unsure		= 5;
	            
				job.onResult = function()
				{
					this.target.checkingConnection = false;

					if( !this.result[0] )
					{
						if( this.unsure > 0 && this.result[1] )
						{
							//
							// target has not completely quitted yet,
							// give it another chance
							//
							this.unsure--;
							
							$.sleep(500);
							this.target.checkConnection( true );
						}
						else
						{
							if( !this.force && this.target.isDebugging() )
							{
								//
								// target is actually in debug mode, 
								// give it a second chance
								//
								$.sleep(500);
								this.target.checkConnection( true );
							}
							else
								this.target.disconnect( true );
						}
					}
				}
	                
				job.onError = job.onTimeout = function()
				{
					this.target.checkingConnection = false;
					this.target.disconnect( true );
				}
	            
				job.submit(true);
			}
			catch( exc )
			{
				this.checkingConnection = false;
				this.disconnect( true );
			}
		}
	}
}

//-----------------------------------------------------------------------------
// 
// setChangeConnectionState(...)
// 
// Purpose: get/set changing connection state
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.setChangeConnectState = function( change )
{
	var notify = ( this.changeConnectState != change );
    this.changeConnectState = change;
	
	if( notify )
		this.mgr.notifyClients( 'changingConnectionState', this );
}

TargetInfo.prototype.getChangeConnectState = function()
{
    return this.changeConnectState;
}

//-----------------------------------------------------------------------------
// 
// canDebug(...)
// 
// Purpose: Can the given session be debugged
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.canDebug = function( session )
{
    return ( this.getConnected() && session && this.exists( session ) && !session.isDebugging() );
}

//-----------------------------------------------------------------------------
// 
// isDebugging(...)
// 
// Purpose: Is one or more of my session currently in debug mode
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.isDebugging = function()
{
    var isDebugging = false;
    
    for( var i=0; i<this.sessions.length; i++ )
        isDebugging = isDebugging || this.sessions[i].isDebugging();
        
    return isDebugging;
}

//-----------------------------------------------------------------------------
// 
// getFeature(...)
// 
// Purpose: get feature of target or session, if given
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.getFeature = function( feature, session )
{
    var ret = false;
    
    var featureID = parseInt( feature, 10 );

    if( !isNaN( featureID ) )
    {
        if( session && session.initialized() )
        {
            var featureSet = session.sessionObj.features[featureID];
            ret = session.sessionObj.features[featureID].supported;
        }
        else if( this.cdic )
            ret = this.cdic.features[featureID].supported;
    }
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// addFiletypes(...)
// 
// Purpose: Add file types for langID
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.addFiletypes = function( langID )
{
    var fileExts = lang.getFileExtensions( langID );
    
    for( var i=0; i<fileExts.length; i++ )
        this.addFiletype( fileExts[i] );
}

//-----------------------------------------------------------------------------
// 
// addFiletype(...)
// 
// Purpose: Add file type
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.addFiletype = function( ext )
{
    for( var j=0; j<this.fileType.length; j++ )
        if( ext == this.fileType[j] )
            return;
        
    this.fileType.push( ext );
}

//-----------------------------------------------------------------------------
// 
// isExec(...)
// 
// Purpose: is given file type executable
// 
//-----------------------------------------------------------------------------

TargetInfo.prototype.isExec = function( langID )
{
    var fileExts = null;
    if(lang)
        fileExts = lang.getFileExtensions( langID );
    
    //
    // find matches
    //
    for( var i=0; i<this.fileType.length; i++ )
    {
        for( var j=0; j<fileExts.length; j++ )
        {
            if( this.fileType[i] == fileExts[j] )
                return true;
        }
    }
    
    return false;
}
