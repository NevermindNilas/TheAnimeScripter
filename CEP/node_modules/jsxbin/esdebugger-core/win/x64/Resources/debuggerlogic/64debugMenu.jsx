/**************************************************************************
*
*  @@@BUILDINFO@@@ 64debugMenu-2.jsx 3.5.0.17	16-March-2009
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

function DebugMenu()
{
    this.debugLevel = ExecutionContext.DBGLEVEL_BREAK;
    this.dontBreakOnErrors = true;
	broadcaster.registerClient( this, 'run' );
    broadcaster.registerClient( this, 'run_target_started_session' );
	broadcaster.registerClient( this, 'continue' );
	broadcaster.registerClient( this, 'stop' );
	broadcaster.registerClient( this, 'pause' );
	broadcaster.registerClient( this, 'step_into' );
	broadcaster.registerClient( this, 'step_out' );
	broadcaster.registerClient( this, 'step_over' );
	broadcaster.registerClient( this, 'reset' );
	broadcaster.registerClient( this, 'switch_frame' );
    broadcaster.registerClient( this, 'get_frame' );
	broadcaster.registerClient( this, 'eval_script' );
    broadcaster.registerClient( this, 'variable_request' );
    broadcaster.registerClient( this, 'set_variable' );
    broadcaster.registerClient( this, 'pump' );
}

DebugMenu.prototype.onNotify = function( reason, param01, param02, param03 )
{
    switch( reason )
    {
        case 'run':
        case 'continue':
        	this.run(param01, param02);
        	break;
        case 'run_target_started_session':
            this.runTargetStartedSession(param01, param02);
        break;
        case 'stop':
        	this.stop();
        	break;
        case 'pause':
        	this.pause();
        	break;
    	case 'step_into':
	    	this.stepInto();
	    	break;
    	case 'step_out':
	    	this.stepOut();
	    	break;
    	case 'step_over':
	    	this.stepOver();
	    	break;
	    case 'reset':
	    	this.reset();
	    	break;
    	case 'switch_frame':
	    	this.switchFrame(param01);
	    	break;
        case 'get_frame':
            this.getFrame(param01);
        break;
	    case 'eval_script':
	    	this.evalScript(param01);
	    	break;
        case 'variable_request':
            this.sendVariableRequest(param01, param02, param03);
            break;
        case 'set_variable':
            this.setVariableValue(param01, param02);
        break;
        case 'pump':
            cdi.pump();
        break;
    }
}

DebugMenu.prototype.runTargetStartedSession = function(param01, param02)
{
    if(param01 != undefined)
        this.dontBreakOnErrors = param01;
    if(param02 != undefined)
        this.debugLevel = param02;

    if(document)
    {
        document.setCurrentSession();
    }

    var session = document ? document.getCurrentSession() : null;
    if( session )
    {
        session.setState(DebugSession.RUNNING);
        session.finalized = false;
        while(session.finalized == false)
        {
            cdi.pump();
            $.sleep (10);
        }
    }
}

DebugMenu.prototype.run = function(param01, param02)
{
    if(param01 != undefined)
        this.dontBreakOnErrors = param01;
    if(param02 != undefined)
        this.debugLevel = param02;

    var session = document ? document.getCurrentSession() : null;
    if( session && session.state != DebugSession.INACTIVE )
        session.command( DebugSession.CMD_CONTINUE );
    else if( document )
        DebugSession.prepareSession( document.getCurrentTarget(), 
                                     session, 
                                     this.debugLevel,
                                     document, 
                                     true );
}

DebugMenu.prototype.pause = function()
{
    var session = document ? document.getCurrentSession() : null;
        
    if( session && session.isDebugging() )
    {       
	    session.command( DebugSession.CMD_PAUSE );
    }
}

DebugMenu.prototype.stop = function()
{
    var session = document ? document.getCurrentSession() : null;
        
    if( session && session.isDebugging() )
    {
        session.stop();
    }
    document = null;
    documents = [];
}

DebugMenu.prototype.stepInto = function()
{
    var session = document ? document.getCurrentSession() : null;
        
    if( session && session.state != DebugSession.INACTIVE )
    {
	    session.command( DebugSession.CMD_STEPINTO );
	}
    else if( document )
        DebugSession.prepareSession( document.getCurrentTarget(), 
									 session, 
									 ExecutionContext.DBGLEVEL_BREAKIMMIDEATE, 
									 document, 
									 true );
}

DebugMenu.prototype.stepOut = function()
{
    var session = document ? document.getCurrentSession() : null;
        
    if( session && session.state != DebugSession.INACTIVE )
    {       
	    session.command( DebugSession.CMD_STEPOUT );
	}
}

DebugMenu.prototype.stepOver = function()
{
    var session = document ? document.getCurrentSession() : null;
        
    if( session && session.state != DebugSession.INACTIVE )
    {
	    session.command( DebugSession.CMD_STEPOVER );
	}
    else if( document )
        DebugSession.prepareSession( document.getCurrentTarget(), 
									 session, 
									 ExecutionContext.DBGLEVEL_BREAKIMMIDEATE, 
									 document, 
									 true );
}

DebugMenu.prototype.reset = function()
{
	if( document )
	{
        var currSession = document.getCurrentSession();
        
        if( currSession )
            currSession.resetEngine();
	}
}

DebugMenu.prototype.switchFrame = function(param01)
{
	if( document )
	{
        var currSession = document.getCurrentSession();
        
        if( currSession ){
            currSession.switchFrame(param01);
        }
	}
}

DebugMenu.prototype.getFrame = function(param01)
{
    if( document )
    {
        var currSession = document.getCurrentSession();
        
        if( currSession ){
            currSession.getFrame(param01);
        }
    }
}

DebugMenu.prototype.evalScript = function(param01)
{
	if( document )
	{
        var currSession = document.getCurrentSession();
        
        if( currSession ){
            currSession.eval(param01);
        }
	}
}

DebugMenu.prototype.sendVariableRequest = function(param01, param02, param03)
{
    if( document )
    {
        var currSession = document.getCurrentSession();

        if( currSession ){
            currSession.sendVariableRequest(param01, param02, param03);
        }
    }

}

DebugMenu.prototype.setVariableValue = function(param01, param02)
{
    if( document )
    {
        var currSession = document.getCurrentSession();

        if( currSession ){
            currSession.setVariableValue(param01, param02);
        }
    }

}

