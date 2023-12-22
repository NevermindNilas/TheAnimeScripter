/**************************************************************************
*
*  @@@BUILDINFO@@@ 00globals-2.jsx 3.5.0.43	20-November-2009
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
// check for BridgeTalk, emulate if not available
//

if( typeof BridgeTalk == 'undefined' )
{
    function BridgeTalk()
    {
        this.sender  = '';
        this.target  = '';
        this.timeout = 0;
        this.type    = '';
        this.body    = '';
        this.headers = {};
    }
    BridgeTalk.prototype.send       = function(){ return false; }
    BridgeTalk.prototype.sendResult = function(){ return false; }
    BridgeTalk.prototype.pump       = function(){ return false; }

    BridgeTalk.launch               = function(){}
    BridgeTalk.isRunning            = function(){ return false; }
    BridgeTalk.pump                 = function(){ return false; }
    BridgeTalk.loadAppScript        = function(){ return false; }
    BridgeTalk.getSpecifier         = function(){ return ''; }
    BridgeTalk.getTargets           = function(){ return []; }
    BridgeTalk.getDisplayName       = function(){ return ''; }
    BridgeTalk.bringToFront         = function(){}
    BridgeTalk.getStatus            = function(){ return ''; }
    BridgeTalk.ping                 = function(){ return ''; }
    BridgeTalk.getAppPath           = function(){ return ''; }
    BridgeTalk.getPreferredApp      = function(){ return ''; }
    BridgeTalk.pendingResponses     = function(){ return 0; }
    BridgeTalk.supportsESTK         = function(){ return false; }
    BridgeTalk.getOMVDictionaryType = function(){ return false; }
    BridgeTalk.getInfo              = function(){ return ''; }
    BridgeTalk.isInstalled          = function(){ return false; }
    BridgeTalk.updateConnectorCache = function(){ return false; }

	BridgeTalk.appSpecifier    = '';
	BridgeTalk.appName         = '';
	BridgeTalk.appVersion      = '';
	BridgeTalk.appLocale       = '';
	BridgeTalk.appInfo         = '';
	BridgeTalk.appInstance     = '';
	BridgeTalk.appStatus       = '';;
	
	BridgeTalk.emu             = true;
}

///////////////////////////////////////////////////////////////////////////////

const _win = (File.fs == "Windows");

// document manager
var docMgr = null;

// The list of all document
var documents = [];

// The current document
var document = null;

// target manager
var targetMgr = null;

// global broadcaster
var globalBroadcaster = new Broadcaster;

// in shutdown
var appInShutDown = false;

var debugMenu = null;

///////////////////////////////////////////////////////////////////
//
//	print()
//
///////////////////////////////////////////////////////////////////

function print()
{
	var s = "";
	for (var i = 0; i < arguments.length; i++)
		s += arguments [i];
	globaljslogger.debugConsoleLog(s);
}

function log()
{
	var s = "";
	for (var i = 0; i < arguments.length; i++)
		s += arguments [i];
	globaljslogger.log(s);
    var traceString = "<trace>";
    traceString += s;
    traceString += "</trace>";
    if(sessionmanagerglo)
        sessionmanagerglo.announceSessionDataAvailable(traceString);
}

//
// wait for BridgeTalk
//
function wait( abortFct, timeout )
{
    var ret = false;
    
    if( abortFct )
    {
        if( !timeout )
            timeout = 50000;
            
	    var then = new Date;
	    
	    while( abortFct() )
	    {
		    BridgeTalk.pump();
		    cdi.pump();
		    
		    var now = new Date;
		    
		    if( (now - then) > timeout )
		        return ret;
	    }
	    
	    ret = true;
	}
	
	return ret;
}


///////////////////////////////////////////////////////////////////////////////
//
// Error Info class
//

function ErrorInfo( error )
{
    this.errors = [];

    if( error )
        this.push( error );
}

ErrorInfo.prototype.push = function( error )
{
    globaljslogger.set_last_error(error);
    log("[" + (new Date()).valueOf() + "] Error: " + error);
    this.errors.push( error );
}

ErrorInfo.prototype.pop = function()
{
    this.errors.pop();
}

ErrorInfo.prototype.length = function()
{
    return this.errors.length;
}

function InternalError( errorInfo )
{
    var error = errorInfo;
    
    if( !error )
        error = new ErrorInfo();
        
    error.push( "InternalError" );
}

///////////////////////////////////////////////////////////////////////////////
//
// string utils
//

function stripWS( str )
{
    var tmp = stripLeadingWS( str );
    tmp     = stripTrailingWS( tmp );
    
    return tmp;
}

function stripLeadingWS( str )
{
    for( var i=0; i<str.length; i++ )
    {
        var ch = str.charAt(i);
        
        if( ch != ' ' && ch != '\t' && ch != '\n' )
        {
            if( i+1 >= str.length )
                return '';
            else
                return str.substring( i+1 );
        }
    }
    
    return str;
}

function stripTrailingWS( str )    
{
    for( var i=str.length-1; i>=0; i-- )
    {
        var ch = str.charAt(i);
        
        if( ch != ' ' && ch != '\t' && ch != '\n' )
        {
            if( i+1 >= str.length )
                return str;
            else
                return str.substr( 0, i+1 );
        }
    }
    
    return str;
}

function createXMLFromString( str )
{
    var ret = null;
    if( str.length > 0 )
    {
        try
        {
            ret = new XML( str );
        }
        catch( exc )
        {
            ret = null;
        }
    }

    if( ret && ret.toXMLString().length <= 0 )
        ret = null;

    return ret;
}
