/**************************************************************************
*
*  @@@BUILDINFO@@@ 80document-2.jsx 3.5.0.25	05-May-2009
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

//-----------------------------------------------------------------------------
// 
// createSourceDocument(...)
// 
// Purpose: factory function
// 
//-----------------------------------------------------------------------------

function createSourceDocument( scriptID, scriptSource, targetSpecifier, engine )
{
    return new SourceDocument( scriptID, scriptSource, targetSpecifier, engine);
}

///////////////////////////////////////////////////////////////////////////////
//
// class SourceDocument
//

//-----------------------------------------------------------------------------
// 
// SourceDocument(...)
// 
// Purpose: ctor
// 
//-----------------------------------------------------------------------------

function SourceDocument( scriptID, scriptSource, targetSpecifier, engine )
{
    this.langID                 = "js";         // the lexer language
	this.isSourceDocument       = true;         // this is a SourceDocument
    this.scriptID               = scriptID;
    this.text                   = scriptSource;
    this.currentTargetSpec      = null;
    this.currentEngine          = null;
    this.currentTarget          = null;
    this.currentSession         = null;
    this.breakpointsArray       = [];

    if(targetSpecifier)
        this.currentTargetSpec = targetSpecifier;

    if(engine)
        this.currentEngine = engine;

    this.setCurrentTarget();
    this.setCurrentSession();
}

SourceDocument.prototype.initialize = function( )
{
    globalBroadcaster.registerClient( this, 'shutdown,addSessionBreakpoints' );
}

SourceDocument.prototype.setText = function( source )
{
    this.text = source;
}

SourceDocument.prototype.getText = function()
{
	return this.text;
}

//-----------------------------------------------------------------------------
// 
// onNotify(...)
// 
// Purpose: notify handler
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.onNotify = function( reason )
{    
    //var currTarget  = this.getCurrentTarget();
    //var currSession = this.getCurrentSession();

    switch( reason )
    {
        case 'addSessionBreakpoints' :
        {
            var sessionAddr = arguments[1];
            var breakpoints = arguments[2];
            
            if( breakpoints )
            {
                for( var i=0; i<this.breakpointsArray.length; i++ )
                    breakpoints.push( this.breakpointsArray[i] );
            }               
        }
        break;

        case 'shutdown':
        {
            try
            {
                globalBroadcaster.unregisterClient( this );
            }
            catch( exc )
            {}
        }
        break;
    }
}

//-----------------------------------------------------------------------------
// 
// getAllBreakpoints(...)
// 
// Purpose: Get a list of all breakpoints. 
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.getAllBreakpoints = function( enabled )
{
    return this.breakpointsArray;
}

//-----------------------------------------------------------------------------
// 
// setAllBreakpoints(...)
// 
// Purpose: set the breakpoints from breakpoints XML
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.setAllBreakpoints = function( breakpointsXML )
{
    //clear the array
    var length = this.breakpointsArray.length;
    for( var i=0; i<length; i++ )
        this.breakpointsArray.pop();

    var xml = createXMLFromString(breakpointsXML);

    if(xml == null)
    {
        log("[" + (new Date()).valueOf() + "] setAllBreakpoints Error: parsing breakpoints XML");
    }

    if( xml.breakpoint && xml.breakpoint.length() > 0 )
    {
        for( var i=0; i<xml.breakpoint.length(); i++ )
        {
            var line = xml.breakpoint[i].@line.toString();
            var enabled = xml.breakpoint[i].@enabled;
            var hits = xml.breakpoint[i].@hits;
            var count = xml.breakpoint[i].@count;
            var condition = xml.breakpoint[i];

            if(hits == undefined)
                hits = 1;
            if(count == undefined)
                count = 0;
            if(condition == undefined)
                condition = "";

            this.breakpointsArray.push(new Breakpoint( line, this.scriptID, (enabled == 1), hits.toString(), count.toString(), condition ));
        }
    }

    targetMgr.sendBreakpoints();
}

//-----------------------------------------------------------------------------
// 
// removeAllBreakpoints(...)
// 
// Purpose: Remove all breakpoints and conditions.
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.removeAllBreakpoints = function()
{
    //clear the array
    var length = this.breakpointsArray.length;
    for( var i=0; i<length; i++ )
        this.breakpointsArray.pop();

	targetMgr.sendBreakpoints();
}

SourceDocument.prototype.setCurrentTarget = function(targetSpecifier)
{
    if(targetSpecifier && targetSpecifier != this.currentTargetSpec)
    {
        this.currentTargetSpec = targetSpecifier;
    }
    
    try
    {
        if( targetMgr ){
            this.currentTarget = targetMgr.findTargetBySpecifier(this.currentTargetSpec);
        }
    }
    catch( exc )
    {
        log("[" + (new Date()).valueOf() + "] setCurrentTarget:findTargetBySpecifier Exception: " + exc.toString());
    }

}

SourceDocument.prototype.setCurrentSession = function(engine)
{
    if(engine && engine != this.currentEngine)
    {
        this.currentEngine = engine;
    }
    
    try
    {
        if( targetMgr ) {
            var sessions = [];
            sessions = targetMgr.findSessionSpecifier(this.currentTargetSpec, this.currentEngine);
            if(sessions.length > 0)
                this.currentSession = sessions[0];
        }
    }
    catch( exc )
    {
        log("[" + (new Date()).valueOf() + "] setCurrentSession:findSessionSpecifier Exception: " + exc.toString());
    }
}

//-----------------------------------------------------------------------------
// 
// getCurrentTarget(...)
// 
// Purpose: Get current target or session
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.getCurrentTarget = function()
{
    return this.currentTarget;
}

SourceDocument.prototype.getCurrentSession = function()
{
    return this.currentSession;
}

//-----------------------------------------------------------------------------
// 
// setLanguage(...)
// 
// Purpose: Set the lexer language.  Also, pass the list of keywords to
//          Auto Completion helper.
// 
//-----------------------------------------------------------------------------

SourceDocument.prototype.setLanguage = function( langID )
{
    if( langID != this.langID )
        this.langID = langID;

    this.status = undefined;
}
