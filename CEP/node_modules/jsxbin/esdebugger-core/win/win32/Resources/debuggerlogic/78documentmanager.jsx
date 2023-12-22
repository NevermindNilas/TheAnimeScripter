/**************************************************************************
*
*  @@@BUILDINFO@@@ 78documentmanager-2.jsx 3.5.0.48	14-December-2009
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
// Document manager
//

//
// Global properties are defined in 00global.jsx:
//
//  document    - The current active document
//  documents   - Array of all opened documents
//

function DocumentManager()
{   
    this.scriptID = 0;		// next document ID for the window
    broadcaster.registerClient( this, 'script_store' );
    broadcaster.registerClient( this, 'activate_target' );
    broadcaster.registerClient( this, 'set_breakpoints' );
}

DocumentManager.prototype.onNotify = function(reason, param01, param02, param03, param04)
{
    switch( reason )
    {
        case 'script_store':
            this.store(param01, param02, param03, param04);
            break;
        case 'activate_target':
            targetInfoBySpecifier = targetMgr.findTargetBySpecifier(param01);
            targetMgr.setActive(targetInfoBySpecifier);
            break;
        case 'set_breakpoints':
            this.setBreakpoints(param01);
            break;
    }
}

//-----------------------------------------------------------------------------
// 
// store(...)
// 
// Purpose: store a new document
// 
//-----------------------------------------------------------------------------

DocumentManager.prototype.store = function( scriptID, scriptSource, targetSpecifier, engine )
{	
    // reset global remote flag
    remoteLaunched = false;
		
    var docObj = this.find(scriptID);

    if(docObj)
    {
        docObj.currentTargetSpec = targetSpecifier;
        docObj.currentEngine = engine;
        docObj.setCurrentTarget();
        docObj.setCurrentSession();
    }

    if(docObj == null)
    {
        try
        {
            docObj = createSourceDocument( scriptID, scriptSource, targetSpecifier, engine);
            docObj.initialize();
            documents.push( docObj );
        }
        catch( exc )
        {
            log("[" + (new Date()).valueOf() + "] store:createSourceDocument Exception: " + exc.toString());
        }
    }

	// update the global document variables
    if(targetSpecifier != "empty" && engine != "empty"){
	   document = docObj;
       var targetInfoBySpecifier = targetMgr.findTargetBySpecifier(targetSpecifier);
       targetMgr.setActive(targetInfoBySpecifier, document.getCurrentSession());
    }
    
}

//-----------------------------------------------------------------------------
//
// setBreakpoints(...)
//
// Purpose: setBreakpoints
//
//-----------------------------------------------------------------------------

DocumentManager.prototype.setBreakpoints = function( breakpointsXML )
{
    //fetch the scipt id from breakpoints xml
    var xml = createXMLFromString(breakpointsXML);

    if(!xml){
        log("[" + (new Date()).valueOf() + "] setBreakpoints Error: breakpoints XML is not a valid XML");
        return;
    }

    var scriptID = xml.@scriptID;

    if(!scriptID)
    {
        log("[" + (new Date()).valueOf() + "] setBreakpoints Error: scriptID is not present in breakpoints XML");
        return;
    }

    if(scriptID == "")
    {
        log("[" + (new Date()).valueOf() + "] setBreakpoints Error: scriptID is empty");
        return;
    }

    //fetch the SourceDocument Object from scriptID
    var docObj = this.find(scriptID);

    if(!docObj)
    {
        log("[" + (new Date()).valueOf() + "] setBreakpoints Error: script not found");
        return;
    }

    //store all breakpoints in SourceDocument Object
    docObj.setAllBreakpoints(breakpointsXML);
}

//-----------------------------------------------------------------------------
// 
// find(...)
// 
// Purpose: Find opened document with given scriptID
// 
//-----------------------------------------------------------------------------

DocumentManager.prototype.find = function( scriptID )
{
    for( var i=0; i<documents.length; i++ )
    {
        if( documents [i].scriptID == scriptID )
            return documents [i];
    }
    
    return null;
}

//-----------------------------------------------------------------------------
// 
// isFile(...)
// 
// Purpose: Is the passed scriptID a file?
// 
//-----------------------------------------------------------------------------

DocumentManager.prototype.isFile = function( scriptID )
{
    // true if the scriptID contains an absolute URI.
    // Do not use File.exists - it may be slow on networked files
    return( scriptID && scriptID.length && ( scriptID[0] == '/' || scriptID[0] == '~' ) );
}
