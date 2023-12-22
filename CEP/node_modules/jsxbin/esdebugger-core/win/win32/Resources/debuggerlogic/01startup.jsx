/**************************************************************************
*
*  @@@BUILDINFO@@@ 01startup-2.jsx 3.0.0.27  22-May-2008
*  ADOBE SYSTEMS INCORPORATED
*  Copyright 2010 Adobe Systems Incorporated
*  All Rights Reserved.
* 
* NOTICE:  Adobe permits you to use, modify, and distribute this file in accordance
* with the terms of the Adobe license agreement accompanying it.  If you have 
* received this file from a source other than Adobe, then your use, modification, or 
* distribution of it requires the prior written permission of Adobe.
**************************************************************************/

// This global variable reflects whether we were launched by another application (via IDEBackend::launchIDE) or not

var remoteLaunched = false;

function app()
{
	broadcaster.registerClient( this, 'app_startup' );
}

app.prototype.onNotify = function(reason)
{
	if( reason == 'app_startup' )
    {
		cdicMgr   = new CDICManager();
	    targetMgr = new TargetManager();
	    docMgr    = new DocumentManager();
	  
	    // init targets
	    targetMgr.loadTargets();
        debugMenu = new DebugMenu();
        targetmanagerglo.registerJSObject(targetMgr);

        //this.remoteLaunched = remoteLaunched;
    }
}

app.prototype.targetAppRunning = function( target )
{
    var ret = false;
    
    if( target.cdic )
    {
        try
        {
            var res = cdicMgr.callSynchronous( target.cdic.isTargetRunning( target.address ) );

            //
            // first res entry is the actual result array, if there're more entries
            // then an error or a timeout occurred
            //
            if( res.length <= 1 )
                ret = res[0][0];
        }
        catch( exc )
        {
            log("[" + (new Date()).valueOf() + "] targetAppRunning Exception: " + exc.toString());
        }
    }
    
    return ret;
}

app.prototype.launchTargetAppSynchronous = function( target, connectIfLaunched, callback )
{
    var ret = false;
    var finished = false;
    
    function check( state )
    {
        ret = state;
        finished = true;
    }
    
    var cb = new Callback( check );
    
    this.launchTargetApp( target, cb );
    
    while( !finished )
        cdi.pump();

    if( ret && connectIfLaunched && target ){
        callback.call(true);
    }

    return ret;
}

app.prototype.launchTargetApp = function( target, callback, errorInfo )
{   
        
    //
    // initiate launch
    //
    if( target.cdic )
    {
        try
        {
            var job         = target.cdic.launchTarget( target.address );
            job.target      = target;
            job.cb          = callback;
            job.errorInfo   = errorInfo;
            
            job.onResult = function()
            {
                if( this.result[0] )
                {
                    //
                    // target app is about to launch, now wait until the app finished launching. timeout after 5 Min.
                    //               
                    const kLaunchTimeout = 120000; // 2 min.
                    var startTime = new Date();
                    var appLaunched = false;
                    var abort = false;

                    while( !appLaunched && !abort )
                    {
                        try
                        {
                            var task = this.target.cdic.isTargetRunning( this.target.address );
                            var res = CDICManager.getSynchronousResult( cdicMgr.callSynchronous( task ) );

                            if( res && res.length && res[0] )
                            {                           
                                appLaunched = true;
                            }

                        }
                        catch( exc )
                        {
                            log("[" + (new Date()).valueOf() + "] launchTargetApp Exception: " + exc.toString());
                        }

                        var now = new Date();

                        abort = ( ( now - startTime ) > kLaunchTimeout ) ;
                    }
                    
                    if( this.cb )
                        this.cb.call( !abort );
                }
                else
                {
                    var error = new ErrorInfo( localize( "$$$/ESToolkit/Alerts/CannotLaunch=Cannot launch target %1", this.target.getTitle()) );
                }
            }
            
            job.onError = job.onTimeout = function()
            {
               var error = new ErrorInfo( localize( "$$$/ESToolkit/Alerts/CannotLaunch=Cannot launch target %1", this.target.getTitle()) );
            }
            
            job.submit(true);
        }
        catch( exc )
        {
             log("[" + (new Date()).valueOf() + "] launchTargetApp Exception: " + exc.toString());
            if( this.cb )
                this.cb.call( false );
        }
    }
    else if( this.cb ){
        log("[" + (new Date()).valueOf() + "] launchTargetApp Error: No cdic.");
        this.cb.call( false );
    }
}

appInstance = new app();

