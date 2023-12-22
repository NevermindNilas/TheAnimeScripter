 /**************************************************************************
*
*  @@@BUILDINFO@@@ 01BridgeTalk-2.jsx 3.5.0.7		08-December-2008
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

//
// clients registered for onReceive
//
BridgeTalk.clients = [];

//
// The BridgeTalk stack contains all active BridgeTalk instances.
//
BridgeTalk.stack = [];

//
// Create a new BridgeTalk instance.
//
BridgeTalk.createXML = function( target, xml, handleTimeout )
{
	XML.prettyPrinting = false;

    var bt  = null;
    
    if( xml )
    {
        bt        = new BridgeTalk;
        bt.type   = "Debug";
        bt.sender = "estoolkit-4.0#dbg";
        bt.target = target + "#estk";
        bt.body   = ( typeof(xml) == "xml" ? xml.toXMLString() : xml );
        
        BridgeTalk.stack.push (bt);

        if( handleTimeout )
            bt.onTimeout = bt.timeoutHandler;

    }       
    return bt;
}

BridgeTalk.create = function( target, command, handleTimeout )
{
    return BridgeTalk.createEx( target + "#estk", command, "Debug", handleTimeout );
}

BridgeTalk.createEx = function( target, command, type, handleTimeout )
{
    var bt    = new BridgeTalk;
    bt.type   = type;
    bt.sender = "estoolkit-4.0#dbg";
    bt.target = target;
    if (command != undefined)
	    bt.headers.Command = command;
    bt.headers.ProtocolVersion = $.version.split (' ')[0];	// ignore the (debug) suffix
    
    BridgeTalk.stack.push (bt);

    if( handleTimeout )
        bt.onTimeout = bt.timeoutHandler;
        
    if( type != "ExtendScript"  &&
        type != "Debug"         &&
        type != "Receive"       &&
        type != "Received"      &&
        type != "Result"        &&
        type != "Error"         &&
        type != "Idle"          &&
        type != "Timeout"       &&
        type != "Log"           &&
        type != "Ignore"        &&
        type != "Processed"     &&
        type != "Launched"          )
    {
        var es = "bt.onResult" + type + "=bt.onResult";     
        eval( es );
    }
        
    return bt;
}

//
// The default error and result handlers remove the BridgeTalk instance after use.
//
BridgeTalk.prototype.onError = function (bt)
{
    if( !this.headers.Proccessed )
    {
	    if (this.onOwnError)
		    this.onOwnError (bt);
    		
	    this.headers.Proccessed = true;
	}
	
	this.destroy();
}

BridgeTalk.prototype.timeoutHandler = function (bt)
{
    if( !this.headers.Proccessed )
    {
	    if (this.onOwnTimeout)
		    this.onOwnTimeout(bt);
        else if (this.onOwnError)
		    this.onOwnError (bt);
    		
	    this.headers.Proccessed = true;
    }
    	
	this.destroy();
}

BridgeTalk.prototype.onResult = function (bt)
{
    if( !this.headers.Proccessed )
    {
	    if (this.onOwnResult)
		    this.onOwnResult (bt);
    		
	    this.headers.Proccessed = true;
    }
    	
	this.destroy();
}

//
// Destroy a BridgeTalk instance by removing it from the list of all
// instances. Also set the target to the empty string to indicate its
// "destruction".
//
BridgeTalk.prototype.destroy = function()
{
    var target  = this.target.replace ("#estk", "");
    var command = this.headers.Command;
    
	for( var i=0; i<BridgeTalk.stack.length; i++ )
	{
		if( BridgeTalk.stack[i] == this )
		{
            BridgeTalk.stack[i] = null;		    
			BridgeTalk.stack.splice( i, 1 );
			break;
		}
	}

	this.target = "";
}

//
// Decode the body according to the escaping guidelines and create a two-dimensional
// array. The major array contains each line as an array of elements that were separated
// by commas. Return that array.
//
BridgeTalk.prototype.splitBody = function()
{
	var reply = this.body.split ('\n');
	// forget about an empty line at the end
	if (reply [reply.length-1].length == 0)
		reply.pop();
	for (var i = 0; i < reply.length; i++)
	{
		var line = reply [i];
		reply [i] = [];
		var esc = false;
		var item = "";
		for (var j = 0; j < line.length; j++)
		{
			var ch = line [j];
			if (ch == '\\' && !esc)
				esc = true;
			else if (esc)
			{
				// escaped character
				if (ch == 'n')
					ch = '\n';					
				item += ch;
				esc = false;
			}
			else if (ch == ',')
			{
				// separator found; eval the part to get rid of escapes
				reply [i].push (item);
				item = "";
				// comma at end: push an empty string
				if (j == line.length-1)
					reply [i].push (item);
			}
			else
				item += ch;
		}
		if (item.length)
			reply [i].push (item);
	}
	return reply;
}

//
// Encode a string according to the protocol conventions.
//
BridgeTalk.encode = function (s)
{
	var t = "";
	for (var i = 0; i < s.length; i++)
	{
		switch (s[i])
		{
			case ',':	t += "\\,"; break;
			case '\\':	t += "\\\\"; break;
			case '\n':	t += "\\n"; break;
			default:	t += s [i];
		}
	}
	return t;
}

//
// Send a BT message after making sure that the target is running
//
BridgeTalk.prototype.safeSend = function (syncTimeout)
{
	var target = this.target.replace ("#estk", "");

    //
    // send only if target is launched
    //
    if( BridgeTalk.isRunning( target ) )
        return this.send();
        
    // not sent, destroy message
    this.destroy();
    
    return false;
}

BridgeTalk.registerClient = function( client )
{
    for( var i=0; i<BridgeTalk.clients.length; i++ )
        if( BridgeTalk.clients[i] == client )
            return;
            
    BridgeTalk.clients.push( client );
}

BridgeTalk.unregisterClient = function( client )
{
    for( var i=0; i<BridgeTalk.clients.length; i++ )
        if( BridgeTalk.clients[i] == client )
        {
            BridgeTalk.clients.splice(i,1);
            break;
        }
}

////////////////////////////////////////////////////////////////
//
// The BridgeTalk receive handler dispatches incoming messages
// according to the type. 
//
BridgeTalk.onReceive = function (bt)
{
    for( var i=0; i<BridgeTalk.clients.length; i++ )
    {
        if( BridgeTalk.clients[i].processBridgeTalk )
            BridgeTalk.clients[i].processBridgeTalk(bt);
    }
	
	return '';
}
