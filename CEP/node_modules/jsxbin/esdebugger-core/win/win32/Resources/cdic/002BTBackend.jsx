/**************************************************************************
*
*  @@@BUILDINFO@@@ 11BTBackend-2.jsx 3.5.0.40	02-November-2009
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

BTBackend = {};

//
// backend name
//
BTBackend.name = 'BTBackend';

//
// backend protocol
//
BTBackend.MIXED    = 0;
BTBackend.PUREXML  = 1;
BTBackend.NOXML    = 2;

//
// debugging commands
//
BTBackend.CmdStop       = [ 'Halt',      'halt'     ];
BTBackend.CmdPause      = [ 'Stop',      'break'    ];
BTBackend.CmdStepOver   = [ 'StepOver',  'stepover' ];
BTBackend.CmdStepInto   = [ 'StepInto',  'stepinto' ];
BTBackend.CmdStepOut    = [ 'StepOut',   'stepout'  ];

//
// additional features
//
Feature.COMPATIBILITY_CS3 = 500;
Feature.COMPATIBILITY_CS4 = 501;
Feature.TO_FRONT		  = 502;

//
// register for broadcast messages
//
broadcaster.registerClient( BTBackend, 'startup' );

var dbgLog = false;

//-----------------------------------------------------------------------------
// 
// TargetInfoBTB(...)
// 
// Purpose: TargetInfo data
// 
// Author : pwollek
// 
//-----------------------------------------------------------------------------

function TargetInfoBTB( address, engine, status, protocol, session )
{
	this.address	= address;
	this.engine		= engine;
	this.status		= status;
	this.protocol	= protocol;
	this.session	= session;

	if( !this.address )
		this.address = null;

	if( !this.engine )
	{
		if( this.address )
			this.engine = this.address.engine;
		else
			this.engine = "";
	}

	if( !this.status )
		this.status = 'active';

	if( !this.protocol )
		this.protocol = BTBackend.PUREXML;

	if( !this.session )
		this.session = null;
}

BTBackend.log = function( message )
{
    globalloggerbt.log( message );
    var traceString = "<trace>";
    traceString += message;
    traceString += "</trace>";
    if(sessionmanagerglo)
        sessionmanagerglo.announceSessionDataAvailable(traceString);
}

//-----------------------------------------------------------------------------
// 
// getTargetScriptIDFormatOS(...)
// 
// Purpose: [static] Does the target require any files in the OS file path format?
//					 Pass in a BridgeTalk message that includes headers or, for
//					 the XML based protocol the xml of the BridgeTalk message body.
// 
//-----------------------------------------------------------------------------

BTBackend.getTargetScriptIDFormatOS = function( btMessage_OR_btXML )
{
	var osFormat = false;

	try
	{
		if( typeof btMessage_OR_btXML == "xml" )
		{
			var xml = btMessage_OR_btXML;

			for( var i=0; i<xml.length(); i++ )
			{
				if( xml[i].name() == "format" )
				{
					osFormat = ( xml[i].file == "OS" );
					break;
				}
			}
		}
		else
		{
			var bt = btMessage_OR_btXML;

			if( bt && bt.headers && bt.headers.Format )
			{
				var formats = bt.headers.Format.split(",");

				for( var i=0; i<formats.length; i++ )
				{
					if( formats[i].indexOf( "file:" ) == 0 )
					{
						var fileFormat = formats[i].split(":");
						osFormat = ( fileFormat[1] ? ( fileFormat[1] == "OS" ) : false );
						break;
					}
				}
			}
		}
	}
	catch( exc )
	{}

	return osFormat;
}

//-----------------------------------------------------------------------------
// 
// createAddress(...)
// 
// Purpose: [static] create Address object based on BridgeTalk specifier
// 
//-----------------------------------------------------------------------------

BTBackend.createAddress = function( specifier, engine, label )
{
	// the spec is name_instance-ver-locale#queue
	var bareName		= specifier.split( '#' )[0];
    var specifierParts  = bareName.split( '-' );
    var nameParts       = specifierParts[0].split( '_' );
    var name            = nameParts[0];
    var version         = specifierParts.length > 1 ? specifierParts[1] : '';
    var locale          = specifierParts.length > 2 ? specifierParts[2] : '';
	var instance        = "";

	if( nameParts.length > 1 )
	{
		for( var i=1; i<nameParts.length; i++ )
		{
			instance += nameParts[i];

			if( i < nameParts.length-1 )
				instance += "_";
		}
	}

	// ignore the locale if it is the current locale
	if (locale == $.locale)
		locale = "";

    if( !label || label.length <= 0 )
        label = BridgeTalk.getDisplayName( specifier );

    if( !label || label.length <= 0 )
    {
        label = name;
        
        if( version.length > 0 )
            label += ' ' + version;
        
		if (locale != "" && instance != "")
			label += " (" + locale + ", " + instance + ")";
		else if (locale != "")
			label += " (" + locale + ")";
		else if (instance != "")
			label += " (" + instance + ")";
            
		if( label[0] >= 'a' && label[0] <= 'z' )
			label = label.toUpperCase()[0] + label.substr(1);
    }
    
    if( !engine )
        engine = '';

	if( instance != "" && label.indexOf( instance ) < 0 )
		label += " (" + instance + ")";
        
    return new Address( BTBackend.name, specifier, instance, '', engine, label );
}

//-----------------------------------------------------------------------------
// 
// createSpecifier(...)
// 
// Purpose: [static] create BridgeTalk specifier for address
// 
//-----------------------------------------------------------------------------

BTBackend.createSpecifier = function( address )
{
    var specifier = address.target;
    
    if( address.instance.length > 0 )
    {
        var specParts = specifier.split('-');
        var instance  = '_' + address.instance;

		if( specParts[0].indexOf( instance ) < 0 )
		{
			specifier = specParts[0] + instance;
        
			if( specParts.length > 1 )
				specifier += '-' + specParts[1];
		}
    }
        
    return specifier;
}

//-----------------------------------------------------------------------------
// 
// initiateSession(...)
// 
// Purpose: [static] Request new session object
// 
//-----------------------------------------------------------------------------

BTBackend.initiateSession = function( engineData, cdic )
{
    var ret = false;
    
    if( engineData && engineData.address && cdic )
    {
        try
        {
            var job        = cdic.newSession( engineData.address );
            job.engineData = engineData;
            
            job.onResult = function()
            {
                if( !this.result[0] )
                    this.engineData.session = null;
            }
            
            job.onError = job.onTimeout = function()
            {
                this.engineData.session = null;
            }
            
            ret = job.submit();
        }
        catch( exc )
        {
            engineData.session = null;
            ret                = false;
        }

        //
        // the session object is available right after the submit
        // and before the debugger returned a true as an acception
        // for the new session. The session object doesn't accept
        // any method calls before the debugger returned true, but
        // we can already install all required handler
        //        
        engineData.session = job.sessionObject;
    }
        
    return ret;
}

//-----------------------------------------------------------------------------
// 
// getTargetProtocol(...)
// 
// Purpose: [static] Get target protocol (MIXED, PUREXML, NOXML) 0f passed target
//                   (Synchronous call)
// 
//-----------------------------------------------------------------------------

BTBackend.getTargetProtocol = function( specifier )
{
    // by default a SpiderMonkey target
	var btData = { protocol : BTBackend.NOXML, done : false };

    var bts  = BridgeTalk.create( specifier, "GetInfo" );
    bts.data = btData;
	
    bts.onOwnResult = function (bt)
    {		           
        var specifier = bt.sender.replace( "#estk", "" );
        
        // line 1: the version, line 2: the supported commands
        var features  = bt.body.split( '\n' );
	    
        if( features.length > 1 )
        {
            features = features[1].split( ',' );
	        
            for( var i=0; i<features.length; i++ )
            {
                if( features[i] == 'get-breakpoints' )
                {
                    // no SpiderMonkey target!
                    
                    if( BridgeTalk.getGroupForSpecifier( specifier ) == "cs/3" )
                        this.data.protocol = BTBackend.MIXED;
                    else
                    {
                        this.data.protocol = BTBackend.PUREXML;
                        
                        //
                        // in case the target isn't in XML protocol
                        // yet, we switch over to XML protocol
                        //
                        var xml = <connect/>;
                        var bt  = BridgeTalk.create( specifier, "UseXML" );
                        bt.safeSend();
                    }

					break;
                }
            }
        }
        
        this.data.done = true;
    }
    
    bts.onOwnError = bts.onOwnTimeout = function( bt )
    {
        this.data.done = true;
    }
    
    if( bts.safeSend() )
    {
        const waitspan = 50000;
        var startTime  = new Date();
        var endTime    = startTime;
        
        while( !btData.done && ( endTime - startTime ) < waitspan )
        {
            this.cdi.pumpBridgeTalk();
            $.sleep( 100 );
        }
        
    }
    
    return btData.protocol;
}

//-----------------------------------------------------------------------------
// 
// createXML(...)
// 
// Purpose: [static] Create XML object based on the passed string.
//                   Return null on any errors.
// 
//-----------------------------------------------------------------------------

BTBackend.createXML = function( str )
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

//-----------------------------------------------------------------------------
// 
// returnErrorXML(...)
// 
// Purpose: [static] Return error on passed job. The str should be an xml error
//                   in the format:
//                   <error id="errCode" file="file" line="line">errMsg</error>
// 
//-----------------------------------------------------------------------------

BTBackend.returnErrorXML = function( job, bt )
{
	var str		= bt.body;
    var errCode = -1;
    var errMsg  = "";
    
    try
    {
        var xml = new XML( str );
        
        if( xml )
        {
			if( xml.@id.toString() )
				errCode = parseInt( xml.@id, 10 );
			else
				errCode = parseInt( bt.headers[ 'Error-Code' ], 10 )

            errMsg  = xml.toString();
        }
    }
    catch( exc )
    {}
        
    try
    {
		if( errMsg.length > 0 )
			job.returnError( errCode, errMsg );
		else
			job.returnError();
    }
    catch( exc )
    {}
}

//-----------------------------------------------------------------------------
// 
// print(...)
// 
// Purpose: Print passed text in debugger output
// 
//-----------------------------------------------------------------------------

BTBackend.print = function( cdic, text, noLE )
{
	if( text.length > 0 )
	{
		if( !noLE && text[ text.length-1 ] != '\n' )
			text += '\n';

		try
		{
			cdic.print( text ).submit();
		}
		catch( exc )
		{}
	}
}

//-----------------------------------------------------------------------------
// 
// onNotify(...)
// 
// Purpose: [static] process broadcast messages
// 
//-----------------------------------------------------------------------------

BTBackend.onNotify = function( reason )
{
    if( reason == 'startup' )
    {
        this.cdi = arguments[1];
        var cdic = this.cdi.registerComponent( BTBackend.name );

		//
		// support CS3/CS4 targets (set at cdic.getTargetAdresses(...) )
		//
		cdic.loadCS3Targets = null;
		cdic.loadCS4Targets = null;

        //
        // register for onReceive BridgeTalk messages
        //
        BridgeTalk.registerClient( cdic );
        
        //
        // cache engine data per target application
        //
        cdic.targets = {};  

		//
		// function scanner
		//
		cdic.fs = null;
		broadcaster.registerClient( cdic, 'fnScanFinished' );
        
        //
        // setup CDIC features
        //    
        
        // STEP_OVER not multiple
        cdic.features[ Feature.STEP_OVER ].priority = 2;    // kSingleDefer

        // don't support single breakpoint action
        cdic.features[ Feature.SET_BREAKPOINT ].supported    = false;
        cdic.features[ Feature.REMOVE_BREAKPOINT ].supported = false;
        
        // no need for dis/connect session
        cdic.features[ Feature.CONNECT ].supported    = false;
        cdic.features[ Feature.DISCONNECT ].supported = false;
        
        //-----------------------------------------------------------------------------
        // 
        // getBridgeTalkGroup(...)
        // 
        // Purpose: Get BridgeTalk group for specifier
        // 
        //-----------------------------------------------------------------------------
        
		cdic.getBridgeTalkGroup = function( specifier )
		{
			var group = BridgeTalk.getGroupForSpecifier( specifier );

			if( !group )
				group = BridgeTalk.getGroupForSpecifier( specifier.split('.')[0] + ".*" );

			return group;
		}

        //-----------------------------------------------------------------------------
        // 
        // setupTargetInfo(...)
        // 
        // Purpose: Setup target info for specifier/engine if not available yet
        // 
        //-----------------------------------------------------------------------------
        
		cdic.setupTargetInfo = function( specifier, engine, address, status, protocol, fileOS )
		{
			var ret = undefined;

			if( !this.targets[specifier] )
			{
				this.targets[specifier]								= {};
				this.targets[specifier].__p__						= {};
				this.targets[specifier].__p__.targetprotocol		= false;
				this.targets[specifier].__p__.setuptargetprotocol	= false;
				this.targets[specifier].__p__.connecting			= false;
				this.targets[specifier].__p__.fileOS				= fileOS;
			}

			if( engine )
			{
				if( !this.targets[specifier][engine] )          
					this.targets[specifier][engine] = new TargetInfoBTB( address, engine, status, protocol );

				ret = this.targets[specifier][engine];
			}
			else
				ret = this.targets[specifier];

			return ret;
		}

        //-----------------------------------------------------------------------------
        // 
        // getTargetInfo(...)
        // 
        // Purpose: Return target info for specifier/engine if available, otherwise return undefined
        // 
        //-----------------------------------------------------------------------------
        
		cdic.getTargetInfo = function( specifier, engine )
		{
			var ret = undefined;

			if( specifier )
			{
				ret = this.targets[specifier];

				if( ret && engine )
					ret = this.targets[specifier][engine];
			}

			return ret;
		}

        //-----------------------------------------------------------------------------
        // 
        // processEnginesXML(...)
        // 
        // Purpose: This function takes XML in the form:
        //          <engines>
	    //              <engine name="name" state="state"/>
	    //              ...more engine names...
        //          </engines>
        //          and store all engine information in the local cache.
        //          Optional all new addresses could be sent to the debugger.
        //          Optional a SessionObject could be created for the first running 
        //          engine or the first of the list.
        // 
        //-----------------------------------------------------------------------------
        
        cdic.processEnginesXML = function( engineXML, specifier, sendAddresses, createSession, fileOS )
        {
            var ret = null;

            this.setupTargetInfo( specifier );
			this.targets[specifier].__p__.fileOS = fileOS;
                
            var displayName     = BridgeTalk.getDisplayName( specifier );
            var addresses       = [];
            var sessionAddr     = null;
            var sessionStatus   = '';
            var firstAddr       = null;
            var firstStatus     = '';
            
            for( var i=0; i<engineXML.engine.length(); i++ )
            {
                var engine = engineXML.engine[i].@name;
                var status = engineXML.engine[i].@state;
                
                //
                // create session address for target/engine
                //
                var addr = BTBackend.createAddress( specifier, engine, displayName );
                addresses.push( addr );	

                //
                // remember running engine for creation of SessionObject
                //
                if( status == "running" )
                {
                    sessionAddr     = addr;
                    sessionStatus   = status;
                }
                
                //
                // remember first address
                //
                if( !firstAddr )
                {
                    firstAddr     = addr;
                    firstStatus   = status;
                }

                //
                // add engine to local cache
                // 	      
				this.setupTargetInfo( specifier, engine, addr, status, BTBackend.PUREXML )
            }
            
            //
            // send addresses to debugger
            //
            if( sendAddresses )
            {
                try
                {
                    var job = this.newAddresses( addresses );
                    job.submit();
                }
                catch( exc )
                {}
            }

            //
            // create SessionObject
            //
            if( createSession )
            {
                if( !sessionAddr )
                {
                    sessionAddr     = firstAddr;
                    sessionStatus   = firstStatus;
                }

                var engineName = sessionAddr.engine;
                var engineData = this.getTargetInfo( specifier, engineName );

                if( engineData && !engineData.session && BTBackend.initiateSession( engineData, this ) )
                {
                    //
                    // setup session object
                    //
                    if( engineData.session )
                        BTBackend.setupSession( engineData.session, sessionStatus, BTBackend.PUREXML  );

                    //
                    // request target protocol
                    //                        
                    this.setupTargetProtocol( specifier );
                    
                    ret = engineData.session;
                }
            }
            else
            {
                //
                // check features of targets
                //
                cdic.setupTargetProtocol( specifier );
            }
            
            return ret;
        }
        
        //-----------------------------------------------------------------------------
        // 
        // findEngineData(...)
        // 
        // Purpose: Attempt to find cached data for passed specifier&engine
        //          If not found then create & initialize data object
        // 
        //-----------------------------------------------------------------------------
        
        cdic.findEngineData = function( specifier, engine )
        {
            var engineData = this.setupTargetInfo( specifier, engine, BTBackend.createAddress( specifier, engine ), 'active', BTBackend.PUREXML );
            
            if( engineData )
            {
                if( engineData.session || ( !engineData.session && BTBackend.initiateSession( engineData, this ) ) )
                {
                    //
                    // setup session object
                    //
                    if( engineData.session && !engineData.session.initialized )
                        BTBackend.setupSession( engineData.session, engineData.status, BTBackend.PUREXML );

                    //
                    // request target protocol
                    //                        
                    this.setupTargetProtocol( specifier, undefined, true );
                }
            }
            
            return engineData;
        }
        
        //-----------------------------------------------------------------------------
        // 
        // setupTargetProtocol(...)
        // 
        // Purpose: Define the target protocol (use no XML or only XML or a mix of both
        //          in BridgeTalk messages) by requesting the Backend feature set
        // 
        //-----------------------------------------------------------------------------
        
        cdic.setupTargetProtocol = function( specifier, job, synchronous )
        {
            if( this.targets[specifier] && !this.targets[specifier].__p__.targetprotocol && !this.targets[specifier].__p__.setuptargetprotocol )
            {
				this.targets[specifier].__p__.setuptargetprotocol = true;

                if( synchronous )
                {
                    var protocol = BTBackend.getTargetProtocol( specifier );

                    //
                    // Set protocol for each engine of the target in the local cache
                    // If a session object is already availabe then set new protocol
                    // at session object
                    //
                    for( var i in this.targets[specifier] )
                    {
						if( i != "__p__" )
						{
							if( this.targets[specifier][i].session && this.targets[specifier][i].session.setProtocol )
								this.targets[specifier][i].session.setProtocol( protocol );
	                            
							this.targets[specifier][i].protocol = protocol;
						}
                    }
                    
                    this.targets[specifier].__p__.targetprotocol = true;
                    
					try
					{
						if( job ) job.returnResult( true );
					}
					catch( exc )
					{}
                }
                else
                {
                    //
                    // check features of targets
                    //
                    var bts  = BridgeTalk.create( specifier, "GetInfo" );
                    bts.job  = job;
                    bts.cdic = this;
                	
                    bts.onOwnResult = function (bt)
                    {		            
                        var specifier = bt.sender.replace( "#estk", "" );
                        
                        if( this.cdic.getTargetInfo( specifier ) )
                        {
                            // by default a SpiderMonkey target                        
							var protocol = BTBackend.NOXML;
                            
                            // line 1: the version, line 2: the supported commands
                            var features  = bt.body.split( '\n' );
                		    
                            if( features.length > 1 )
                            {
                                features = features[1].split( ',' );
                		        
                                for( var i=0; i<features.length; i++ )
                                {
                                    if( features[i] == 'get-breakpoints' )
                                    {
                                        // no SpiderMonkey target!
                                        
                                        if( this.cdic.getBridgeTalkGroup( specifier ) == "cs/3" )
                                            protocol = BTBackend.MIXED;
                                        else
                                        {
                                            protocol = BTBackend.PUREXML;
                                            
                                            //
                                            // in case the target isn't in XML protocol
                                            // yet, we switch over to XML protocol
                                            //
                                            var xml = <connect/>;
                                            var bt  = BridgeTalk.create( specifier, "UseXML" );
                                            bt.safeSend();
                                        }

										break;
                                    }
                                }
                            }
                            
                            //
                            // Set protocol for each engine of the target in the local cache
                            // If a session object is already availabe then set new protocol
                            // at session object
                            //
                            for( var i in this.cdic.targets[specifier] )
                            {
								if( i != "__p__" )
								{
									if( this.cdic.targets[specifier][i].session )
										this.cdic.targets[specifier][i].session.setProtocol( protocol );
	                                    
									this.cdic.targets[specifier][i].protocol = protocol;
								}
                            }
                            
                            this.cdic.targets[specifier].__p__.targetprotocol = true;
                        }
                        
						try
						{
							if( this.job ) this.job.returnResult( true );
						}
						catch( exc )
						{}
                    }
                    
                    bts.onOwnError = function( bt )
                    {
						try
						{
							if( this.job )
							{
								//
								// an error here is ok, 
								// return true for a successfully connection
								//
								this.job.returnResult( true );
							}
						}
						catch( exc )
						{}
                    }
                	
                    bts.safeSend();
                }
            }
        }
        
        //-----------------------------------------------------------------------------
        // 
        // createAddressObject(...)
        // 
        // Purpose: Process Job.CREATE_ADDRESS
        //          Return Address object for specifier
        // 
        //-----------------------------------------------------------------------------

        cdic.createAddressObject = function( job )
        {
            var addr      = null;
            var specifier = BridgeTalk.getSpecifier( job.specifier );
            
            if( null != specifier && specifier.length > 0 )
                addr = BTBackend.createAddress( specifier );
                
			try
			{
				if( addr )
					job.returnResult( addr );
				else
					job.returnError();
			}
			catch( exc )
			{}
        }
        
        //-----------------------------------------------------------------------------
        // 
        // getFileTypes(...)
        // 
        // Purpose: Process Job.GET_FILE_TYPES
        //          Return array of file name extensions
        // 
        //-----------------------------------------------------------------------------

        cdic.getFileTypes = function( job )
        {
			try
			{
				job.returnResult( [ 'js', 'jsx' ] );
			}
			catch( exc )
			{}
        }
        
        //-----------------------------------------------------------------------------
        // 
        // toFront(...)
        // 
        // Purpose: Process Job.TO_FRONT
        //          Try to bring debugger app to front
        // 
        //-----------------------------------------------------------------------------

        cdic.toFront = function( job )
        {
			try
			{
				if( File.fs == "Windows" )
				{
					var bt = BridgeTalk.create( BTBackend.createSpecifier( job.address ), "ToFront" );
					bt.headers.ID = job.hostID;
			        
					job.returnResult( bt.safeSend() );
				}
				else
					// Non-Windows: signal app to use BridgeTalk
					job.returnResult(false);
			}
			catch( exc )
			{}
        }
        
        //-----------------------------------------------------------------------------
        // 
        // getTargetAdresses(...)
        // 
        // Purpose: Process Job.TARGET_ADDRESSES
        //          Get all available target addresses
        // 
        //-----------------------------------------------------------------------------

		cdic.getTargetAdresses = function( job )
        {
			this.loadCS3Targets = this.features[Feature.COMPATIBILITY_CS3].supported;
			this.loadCS4Targets = this.features[Feature.COMPATIBILITY_CS4].supported;

			var addresses = [];

            var apps = null;

			//
            // first get all specifiers
            //
	        apps = BridgeTalk.getTargets( -999, null );

	        //
            // determine for each app if version and/or locale info needs to be added
            // collect this stuff into the following object, create subobjects holding
            // booleans for versions and locales needed
	        //
            var testObj = {};

            for( var pass=1; pass<=2; pass++ )
            {
	            for( var i=0; i<apps.length; i++ )
	            {
					var btGroup = this.getBridgeTalkGroup( apps[i] );

					if( ( btGroup != "cs/3" && btGroup != "cs/4" )	||
						( this.loadCS3Targets && btGroup == "cs/3" )			||
						( this.loadCS4Targets && btGroup == "cs/4" )				)
					{
						// the target is now fully qualified
						var app = ( ( this.loadCS3Targets || this.loadCS4Targets ) ? apps[i] : BridgeTalk.getSpecifier( apps[i] ) );

						if( app && BridgeTalk.supportsESTK( app ) )
						{						
							var target  = app.split ('-');
							var version = target [1];
							var locale  = target [2];
							target      = target [0];

							// Ignore Acrobat 7 for now
							if (target == "acrobat" && Number (version) < 8)
								continue;
							// Ignore the Help Center for now
							if (target == "helpcenter")
								continue;
							// do not load info about myself
							if (target != "estoolkit")
							{
								var appObj = testObj [target];
								if (!appObj)
									appObj = testObj [target] = { needVersion:false, 
																  needLocale:false, 
																  version:version, 
																  locale:locale };
	        					                                  
								if (pass == 1)
								{
									// pass 1: collect information
									if (appObj.version != version)
										appObj.needVersion = true;
									if (appObj.locale != locale)
										appObj.needLocale = true;
								}
								else
								{
									// pass 2: generate the entries
									var displayName = BridgeTalk.getDisplayName (app);
	        					    
									// OK, the dislay name is valid - we can use it
									if (displayName)
									{
										// need to add version and/or locale?
										if( appObj.needVersion && appObj.needLocale && version && locale )
											displayName += " (" + version + ", " + locale + ")";
										else if( appObj.needVersion && version )
											displayName += " (" + version + ")";
										else if( appObj.needLocale && locale )
											displayName += " (" + locale + ")";
	        							    
										var addr = BTBackend.createAddress( app, '', displayName );
										addr.script = btGroup;
										addresses.push( addr );			
									}
								}
							}
						}
					}
	            }
            }
        
			try
			{
    			//
    			// return array of target addresses
    			//
				job.returnResult( addresses );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // getSessionAddresses(...)
        // 
        // Purpose: Process Job.SESSION_ADDRESSES
        //          Get all available session addresses of a target
        // 
        //-----------------------------------------------------------------------------

        cdic.getSessionAddresses = function( job )
        {
            var ret     = [];
            var engines = this.getTargetInfo( BTBackend.createSpecifier( job.address ) );
            
            if( engines )
            {
                for( var i in engines )
                {
					if( i != "__p__" )
					{
						if( engines[i].engine )
						{
							var addr = new Address( job.address.type,
													job.address.target,
													job.address.instance,
													job.address.script,
													engines[i].engine,
													job.address.label );
													
							ret.push( addr );
						}
					}
                }
            }
            
			try
			{
				//
				// return array of session addresses
				//
				job.returnResult( ret );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // launch(...)
        // 
        // Purpose: Process Job.LAUNCH
        //          Launch target application
        // 
        //-----------------------------------------------------------------------------

        cdic.launch = function( job )
        {
            var res = BridgeTalk.launch( BTBackend.createSpecifier( job.address ) );
            
			try
			{
				job.returnResult( res );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // isRunning(...)
        // 
        // Purpose: Process Job.IS_RUNNING
        //          Is target application running?
        // 
        //-----------------------------------------------------------------------------

        cdic.isRunning = function( job )
        {
			if( dbgLog )
				start = new Date();

            var targetSpec  = BTBackend.createSpecifier( job.address );
            var backendSpec = targetSpec + '#estk';
            
            var res1 = BridgeTalk.isRunning( targetSpec );
            var res2 = BridgeTalk.isRunning( backendSpec );

			try
			{
				job.returnResult( ( res1 && res2 ), ( res1 != res2 ) );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // initTarget(...)
        // 
        // Purpose: Process Job.INIT
        //          Initialize target. Before any target- or session related 
        //          functionality is called the debugger initialize the target
        //          (ALWAYS TRY THE XML VERSION FIRST, IF IT FAILS THEN USE NON_XML)
        // 
        //-----------------------------------------------------------------------------

        cdic.initTarget = function( job )
        {
            var specifier = BTBackend.createSpecifier( job.address );
            if( BridgeTalk.isRunning( specifier ) )
                this.initTargetXML( job, specifier );
            else
			{
				try
				{
					job.returnError();
				}
				catch( exc )
				{}
			}
        }
        
        cdic.initTargetNoXML = function( job, specifier )
        {
            //
            // try to connect to target
            //
            var bt          = BridgeTalk.create( specifier, "Connect", true );
            bt.cdic         = this;
            bt.job          = job;
            bt.address      = job.address;
            
            bt.onOwnError = function( bt )
            {
				try 
				{
					this.job.returnError();
				} 
				catch( exc ) 
				{}
            }
            
            bt.onOwnResult = function( bt )
            {	
				var spidermonkey = BTBackend.getTargetScriptIDFormatOS( bt );
                var specifier = bt.sender.replace ("#estk", "");
                var reply       = bt.splitBody();

                this.cdic.setupTargetInfo( specifier, undefined, undefined, undefined, undefined, spidermonkey );

                for( var i=0; i<reply.length; i++ )
                {
                    if( reply [i].length == 0 )
                        // list of supported commands
                        break;
            			
                    var engine = reply [i][0];
                    var status = reply [i][1];
					
					if (!status || status == "true")
						status = "active";

                    //
                    // add engine to local cache
                    // 	      
					this.cdic.setupTargetInfo( specifier, engine, null, status, BTBackend.PUREXML );
                }
                
                //
                // check features of targets (also quit the job)
                //
                this.cdic.setupTargetProtocol( specifier, this.job );

				try
				{
					this.job.returnResult( true );
				}
				catch( exc )
				{}
			}
       	    
            if( !bt.safeSend() )
                job.returnError();
        }
        
        cdic.initTargetXML = function( job, specifier )
        {
			//
            // try to connect to target
            //
            var xml = <connect/>;
            var bt  = BridgeTalk.createXML( specifier, xml, true );
            bt.job  = job;
            bt.cdic = this;

            bt.onOwnError = function( bt )
            {
                var specifier = BTBackend.createSpecifier( this.job.address );
                this.cdic.initTargetNoXML( this.job, specifier );
            }
            
            bt.onOwnTimeout = function( bt )
            {
                BTBackend.returnErrorXML( this.job, bt );
            }
            
            bt.onOwnResult = function( bt )
            {
                var xml = BTBackend.createXML( bt.body );
                if( xml )
                {
					var spidermonkey = BTBackend.getTargetScriptIDFormatOS( bt );
					this.cdic.processEnginesXML( xml, 
												 BTBackend.createSpecifier( this.job.address ), 
												 undefined, 
												 undefined, 
												 spidermonkey );

					try
					{
						this.job.returnResult( true );
					}
					catch( exc )
					{}
                }
                else
				{
					try
					{
						this.job.returnError();
					}
					catch( exc )
					{}
				}
            }
       	    
            if( !bt.safeSend() )
                job.returnError();
        }
        
        //-----------------------------------------------------------------------------
        // 
        // exitTarget(...)
        // 
        // Purpose: Job.EXIT
        //          Finalize target. When the debugger finished any interaction with
        //          the target an EXIT is called.
        // 
        //-----------------------------------------------------------------------------

        cdic.exitTarget = function( job )
        {
			try
			{
				this.targets[BTBackend.createSpecifier( job.address )]  = null;
	            
				job.returnResult( true );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // acquireSession(...)
        // 
        // Purpose: Process Job.ACQUIRE_SESSION
        //          Create new session. The debugger requests a new session for the
        //          passed session address.
        // 
        //-----------------------------------------------------------------------------

        cdic.acquireSession = function( job )
        {    
            var ret     = false;

            if( job.address && job.address.engine.length > 0 )
            {
                //
                // the session object is available as soon as
                // one requested a new session.
                // The session object denies any method calls as
                // long as the request has finished, but all
                // required handler could be installed now.
                //                
                var session = job.sessionObject;    //this.cdi.sessions[job.session];
                
                if( session )
                {
                    var specifier = BTBackend.createSpecifier( job.address );
                    var target    = this.setupTargetInfo(specifier);
                        
                    if( target )
                    {
                        var engine = target[job.address.engine];

                        if( !engine )
                        {                    
                            var targetProtocol = BTBackend.getTargetProtocol( specifier );                        
                            
                            //
                            // engine of requested session not cached yet
                            // 
							this.setupTargetInfo( specifier, job.address.engine, job.address, 'exec', targetProtocol );
                            engine = target[job.address.engine];
                        }
                        
                        if( engine )         
                        {
                            //
                            // setup session object
                            //
                            engine.session = session;
                            BTBackend.setupSession( session, engine.status, BTBackend.PUREXML );
                            ret = true;
                        }
                    }
                }
            }
        
			try
			{
				//
				// return true to accept the request
				//
				job.returnResult( ret );
			}
			catch( exc )
			{}
        }

        //-----------------------------------------------------------------------------
        // 
        // getDictionaries(...)
        // 
        // Purpose: Process Job.GET_DICTIONARIES
        //          Get list of dictionaries of target
        // 
        //-----------------------------------------------------------------------------

        cdic.getDictionaries = function( job )
        {        
			var target	= BTBackend.createSpecifier( job.address );
			var bt		= BridgeTalk.create( target);
            bt.job		= job;
			bt.body		= "<get-dictionaries/>";
 
            bt.onOwnResult = function( bt )
            {
				try
				{
					this.job.returnResult( bt.body );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = function(bt)
            {
                BTBackend.returnErrorXML( this.job, bt );
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }

        //-----------------------------------------------------------------------------
        // 
        // getDictTOC(...)
        // 
        // Purpose: Process Job.GET_DICT_TOC
        //          Get table of contents of dictionary
        // 
        //-----------------------------------------------------------------------------

        cdic.getDictTOC = function( job )
        {
			var target	= BTBackend.createSpecifier( job.address );
			var bt      = BridgeTalk.create( target );
            bt.job      = job;
            bt.body     = '<GetTOC prefix="' + 
                          ( job.prefix.length > 0 ? job.prefix : '#/' ) + 
                          '" dictname="' + job.dictionary + '"/>';

            bt.onOwnResult = function( bt )
            {
				try
				{
					this.job.returnResult( bt.body );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = function(bt)
            {
                BTBackend.returnErrorXML( this.job, bt );
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }

        //-----------------------------------------------------------------------------
        // 
        // getClassInfo(...)
        // 
        // Purpose: Process Job.GET_DICT_CLASS
        //          Get info for class of dictionary
        // 
        //-----------------------------------------------------------------------------

        cdic.getClassInfo = function( job )
        {
			var target	= BTBackend.createSpecifier( job.address );
			var bt      = BridgeTalk.create( target );
            bt.job      = job;
            bt.body     = '<GetClassInfo prefix="' +
                          ( job.prefix.length > 0 ? job.prefix : '#/' ) + 
                          '" dictname="' + job.dictionary + '">' + 
                          ( job.group.length > 0 ? ('<element>' + job.group + '</element></GetClassInfo>') : '</GetClassInfo>' );
        	
            bt.onOwnResult = function( bt )
            {
				try
				{
					this.job.returnResult( bt.body );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = function(bt)
            {
                BTBackend.returnErrorXML( this.job, bt );
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }
        
        //-----------------------------------------------------------------------------
        // 
        // scanForFunctions(...)
        // 
        // Purpose: Process Job.SCAN_FUNCTIONS
        //          Scan passed source code string for function definitions
        // 
        //-----------------------------------------------------------------------------

		cdic.onNotify = function( reason )
		{
			if( reason == 'fnScanFinished' )
			{
				try
				{
					if( this.fs && this.fs.ident = arguments[1] )
						this.fs.returnResult( arguments[2] );
				}
				catch( exc )
				{}
			}
		}

        cdic.scanForFunctions = function( job )
        {
			this.fs = job;
			this.scanFunctions( job.source,
								job.ident,
								"function\\s+([$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*)\\s*\\(\\s*([$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*(?:\\s*,\\s*[$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*)*\\s*)?\\)",
								"([$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*)\\s*=\\s*function\\s*\\(\\s*([$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*(?:\\s*,\\s*[$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*)*)?\\s*\\)",
								"([$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*)\\s*=\\s*new\\s+Function\\s*\\(s*((?:\"[$_a-zA-Z](?:[$_a-zA-Z0-9])*(?:\\.[$_a-zA-Z](?:[$_a-zA-Z0-9])*)*\"\\s*,\\s*)*)\\s*\".*\"\\s*\\)" );
        }

		//-----------------------------------------------------------------------------
		// 
		// onTick(...)
		// 
		// Purpose: Pulse
		// 
		//-----------------------------------------------------------------------------

		cdic.onTick = function()
		{
		}

        //-----------------------------------------------------------------------------
        // 
        // processBridgeTalk(...)
        // 
        // Purpose: Process incomming BridgeTalk messages
        // 
        //-----------------------------------------------------------------------------

        cdic.processBridgeTalk = function( bt )
        {
            if( bt && bt.type == 'Debug' )
            {
                if( bt.headers.Command && bt.headers.Command.length > 0 )
                {
					var spidermonkey = BTBackend.getTargetScriptIDFormatOS( bt );

                    switch( bt.headers.Command )
                    {
						case "Print":
						{
							//
							// print string in the debugger output
							//
                            sessionmanagerglo.announceSessionDataAvailable(bt.body);
							BTBackend.print( this, bt.body, true );
						}
						break;

						case "Open":
	                        // sent by a second instance of the Toolkit
	                        // so bring this app to front in any case
	                        if( bt.body.length )
	                            this.openDocument( bt.body ).submit();
	                        break;

                        case "ConnectRequest":
                        {            
							//
							// parameters
							//
							var f = new File( bt.headers.Parameters );

							if( f.exists )
								this.openDocument( bt.headers.Parameters ).submit();

                            //
                            // Sent by a target app wishing to connect
                            //
                            var specifier = bt.sender.replace ("#estk", "");

                            //
                            // If the target is not registered, register it now
                            //
                            this.setupTargetInfo( specifier );
							this.targets[specifier].__p__.fileOS = spidermonkey;
                		    
							if( !this.targets[specifier].__p__.connecting )
							{
								this.targets[specifier].__p__.connecting = true;

								var addresses   = [];
								var sessionAddr = null;
								var sessionStatus = '';
								var reply       = bt.splitBody();
								var selEngine   = bt.headers.Engine;
								var engines     = 0;

								for( var i=0; i<reply.length; i++ )
								{
									if( reply [i].length == 0 )
										// list of supported commands
										break;
	                        			
									var engine = reply [i][0];
									var status = reply [i][1];

									if( !selEngine )
										selEngine = engine;
	                            
									//
									// create session address for target/engine
									//
									var displayName = BridgeTalk.getDisplayName( specifier );
									var addr = BTBackend.createAddress( specifier, engine, displayName );
									addresses.push( addr );	
	                                
									if( engine == bt.headers.Engine )
									{
										sessionAddr = addr;
										sessionStatus = status;
									}

									//
									// add engine to local cache
									//               
									this.setupTargetInfo( specifier, engine, addr, status, BTBackend.PUREXML );
								}

								//
								// send new addresses to debugger
								//
								try
								{
									var job = this.newAddresses( addresses );
									job.target = this.targets[specifier];

									job.onResult = job.onError = job.onTimeout = function()
									{
										this.target.__p__.connecting = false;
									}

									job.submit();
								}
								catch( exc )
								{
									this.targets[specifier].__p__.connecting = false;
								}
	                            
								//
								// create a new session
								//
								if( sessionAddr )
								{
									var specifier  = BTBackend.createSpecifier(sessionAddr);
									var engineName = sessionAddr.engine;
									var engineData = this.getTargetInfo( specifier, engineName );
	                                    
									if( engineData && BTBackend.initiateSession( engineData, this ) )
									{
										//
										// setup session object
										//
										if( engineData.session )
											BTBackend.setupSession( engineData.session, sessionStatus, BTBackend.MIXED );

										//
										// request target protocol
										//                        
										this.setupTargetProtocol( BTBackend.createSpecifier( sessionAddr ) );
									}
								}
							}
                        }    
                        break;
                        
                        default:
                        {                                                
                            //
                            // a session job without session id?
                            //
                            var specifier  = bt.sender.replace ("#estk", "");
                            var engine     = bt.headers.Engine;
                            var engineData = this.findEngineData( specifier, engine );
                            
                            if( engineData )
							{
								this.targets[specifier].__p__.fileOS = spidermonkey;
                                engineData.session.processBridgeTalk( bt );
							}
                        }
                    }
                }
                else
                {				
                    //
                    // XML based messages
                    //
                    var xml = BTBackend.createXML( bt.body );
					var spidermonkey = BTBackend.getTargetScriptIDFormatOS( xml );
                    var cmd = ( ( xml && xml.name() ) ? xml.name().toString() : "" );
        		    
		            if( cmd.length > 0 )
		            {
		                var specifier = bt.sender.replace( '#estk', '' );
		                
		                switch( cmd )
		                {
							case "print":
							{
								//
								// <print engine="name">Text to print</print>
								//

								var text = xml.toString();

								if( text.length <= 0 )
								{
									var start = bt.body.indexOf( ">" ) + 1;
									var end   = bt.body.lastIndexOf( "<" );

									if( start < end )
										text = bt.body.substring( start, end );
								}
                                sessionmanagerglo.announceSessionDataAvailable(bt.body);
								BTBackend.print( this, text, true );
							}
							break;

							//
							// add/remove/modify single engine of target
		                    //
		                    case "engine-name":
		                    {		
		                        var oldName = xml.@old.toString();
		                        var newName = xml.@new.toString();

                                if( oldName.length > 0 || newName.length > 0 )
                                {
                                    if( newName.length <= 0 )
                                    {
                                        //
                                        // remove engine from list
                                        //
                                        if( this.getTargetInfo( specifier, oldName ) )
                                        {
                                           var addr = BTBackend.createAddress( specifier, oldName );
                                           
                                           try
                                           {
                                               this.removeAddress( addr ).submit();
                                           }
                                           catch( exc )
                                           {}
                                           
                                           delete this.targets[specifier][oldName];
                                        }
                                    }
                                    else if( oldName.length <= 0 )
                                    {
										if( this.getTargetInfo(specifier) )
										{
											//
											// add new engine to list
											//
											var addr = BTBackend.createAddress( specifier, newName );
											
											try
											{
												var job = this.newAddresses( addr );
												job.submit();
											}
											catch( exc )
											{}

											//
											// create a new session
											//
											var engineData = this.setupTargetInfo( specifier, addr.engine, addr, 'active', BTBackend.PUREXML );
											
											if( engineData && BTBackend.initiateSession( engineData, this ) )
											{
												//
												// setup session object
												//
												if( engineData.session )
													BTBackend.setupSession( engineData.session, 'dynamic', BTBackend.MIXED );

												//
												// request target protocol
												//                        
												this.setupTargetProtocol( specifier );
											}
										}
                                    }
                                    else
                                    {
                                        //
                                        // change session addr
                                        //
                                        
                                        //
                                        // change local cache
                                        //
                                        if( this.getTargetInfo( specifier, oldName) )
                                        {
                                           this.targets[specifier][newName]        = this.targets[specifier][oldName];
                                           this.targets[specifier][newName].engine = newName;
                                           
                                           delete this.targets[specifier][oldName];
                                        }

                                        //
                                        // anounce change to debugger
                                        //
                                        var oldAddr = BTBackend.createAddress( specifier, oldName );
                                        var newAddr = BTBackend.createAddress( specifier, newName );
                                        
                                        try
                                        {
                                            this.changeAddress( oldAddr, newAddr ).submit();
                                        }
                                        catch( exc )
                                        {
                                            break;
                                        }
                                    }
                                }
		                    }
		                    break;
		                    
		                    //
		                    // target requests connection
		                    //
		                    case "connect-request":
		                    {
								//
                                // Sent by a target app wishing to connect
                                //
                                var sessionObj = null;

		                        if( xml.engines && xml.engines.length() > 0 )
		                            sessionObj = this.processEnginesXML( xml.engines,
		                                                                 specifier, 
		                                                                 true /*send addresses to debugger*/,
		                                                                 true /*create&return SessionObj*/,
																		 spidermonkey );
                                xml.@specifier = specifier;
                                sessionmanagerglo.announceTargetDataAvailable(xml.toString());
		                    }
		                    break;  
		                    
                            default:
                            {                        
                                //
                                // a session job without session id?
                                //
                                var engine     = ( xml.@engine ? xml.@engine.toString() : "" );
                                var engineData = this.findEngineData( specifier, engine );
                                
                                if( engineData )
								{
									this.targets[specifier].__p__.fileOS = spidermonkey;
                                    engineData.session.processBridgeTalk( bt );
								}
                            }
		                }
		            }
                }
            }
			else if( bt && bt.type == 'ExtendScript' )
			{
				var thisObj = this;

				function registerTarget( specifier )
				{
					var addr = BTBackend.createAddress( specifier );

					try
					{
						var job = thisObj.newAddresses( addr );
						job.submit();
					}
					catch( exc )
					{}
				}

				try
				{
					eval( bt.body );
				}
				catch( exc )
				{}

				thisObj = null;
			}
        }

        //-----------------------------------------------------------------------------
        // 
        // onTask(...)
        // 
        // Purpose: Dispatch tasks
        // 
        //-----------------------------------------------------------------------------

        cdic.onTask = function( job )
        {
            switch( job.name )
            {
                case Job.TO_FRONT:
                    this.toFront( job );
                    break;
                    
                case Job.INIT:
                    this.initTarget( job );
                    break;
        	        
                case Job.EXIT:
                    this.exitTarget( job );
                    break;
        	        
    	        case Job.CREATE_ADDRESS:
    	            this.createAddressObject( job );
    	            break;
    	            
    	        case Job.GET_FILE_TYPES:
    	            this.getFileTypes( job );
    	            break;
        	        
                case Job.TARGET_ADDRESSES:
                    this.getTargetAdresses( job );			
	                break;
            		
                case Job.SESSION_ADDRESSES:
                    this.getSessionAddresses( job );
                    break;
        		
                case Job.LAUNCH:
                    this.launch( job );
	                break;
        		
                case Job.IS_RUNNING:
                    this.isRunning( job );
                    break;
    		        
                case Job.ACQUIRE_SESSION:
                    this.acquireSession( job );
                    break;
    		        
                case Job.RELEASE_SESSION:
                    this.releaseSession( job );
                    break;
    	            
                case Job.GET_DICTIONARIES:
                    this.getDictionaries( job );
                    break;
    	            
                case Job.GET_DICT_TOC:
                    this.getDictTOC( job );
                    break;
    	            
                case Job.GET_DICT_CLASS:
                    this.getClassInfo( job );
                    break;
                    
                case Job.SCAN_FUNCTIONS:
                    this.scanForFunctions( job );
                    break;
    	            
                case Job.CUSTOM:
                {
					var command = job.customName;

					switch( command )
					{
						case 'dbgLog':
							dbgLog = job.customArguments[0];
							break;
					}

					try
					{
						job.quitTask();
					}
					catch( exc )
					{}
                }
                break;
    	        
                default:
                {
                    //
                    // job not handled, just quit it
                    //
					try
					{
						job.quitTask();
					}
					catch( exc )
					{}
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//
// BTBackend session
//

//-----------------------------------------------------------------------------
// 
// setupSession(...)
// 
// Purpose: [static] Setup BTBackend session
// 
//-----------------------------------------------------------------------------

BTBackend.setupSession = function( session, status, protocol )
{
    session.protocol        = protocol;
    session.status          = status;
    session.stack           = [];
    session.breakpoints     = [];
    session.dontBreakFlag   = 0;
    session.pendingBTCalls  = [];
    session.pendingXMLCalls = [];
    session.initialized     = true;
    session.profileData     = null;     // cache for profile data (only for protocol PUREXML)

    //-----------------------------------------------------------------------------
    // 
    // print(...)
    // 
    // Purpose: Print passed text in debugger output
    // 
    //-----------------------------------------------------------------------------

    session.print = function( text, noLE )
    {
		if( text.length > 0 )
		{
			if( !noLE, text[ text.length-1 ] != '\n' )
				text += '\n';

			try
			{
				this.cdic.print( text ).submit();
			}
			catch( exc )
			{}
		}
    }

    //-----------------------------------------------------------------------------
    // 
    // setupFeatures(...)
    // 
    // Purpose: Setup session features depending of its protocol
    // 
    //-----------------------------------------------------------------------------

    session.setupFeatures = function()
    {    
	    if( this.protocol == BTBackend.NOXML )
	    {
		    //
		    // SpiderMonkey target
		    //
		    this.features[ Feature.GET_PROFILE_DATA ].supported     = false;
		    this.features[ Feature.DBGFLAG_DONTBREAK ].supported    = false;
		    this.features[ Feature.DBGFLAG_IGNOREERROR ].supported  = false;
		    this.features[ Feature.PROFILE_LINE ].supported         = false;
		    this.features[ Feature.PROFILE_FUNCTION ].supported     = false;
		    this.features[ Feature.PROFILE_TIME ].supported         = false;
		    this.features[ Feature.PROFILE_HITS ].supported         = false;
		    this.features[ Feature.RESET].supported                 = false;
	    }
	    else
	    {
		    //
		    // ExtendScript target
		    //
		    this.features[ Feature.GET_PROFILE_DATA ].supported     = true;
		    this.features[ Feature.DBGFLAG_DONTBREAK ].supported    = true;
		    this.features[ Feature.DBGFLAG_IGNOREERROR ].supported  = true;
		    this.features[ Feature.PROFILE_LINE ].supported         = true;
		    this.features[ Feature.PROFILE_FUNCTION ].supported     = true;
		    this.features[ Feature.PROFILE_TIME ].supported         = true;
		    this.features[ Feature.PROFILE_HITS ].supported         = true;
	    }
    }
    
    //-----------------------------------------------------------------------------
    // 
    // setProtocol(...)
    // 
    // Purpose: Set session protocol
    // 
    //-----------------------------------------------------------------------------

    session.setProtocol = function( protocol )
    {
        var oldProtocol	= this.protocol;
        this.protocol   = protocol;
        
        if( this.protocol != oldProtocol )
            this.setupFeatures();
    }
    
    //-----------------------------------------------------------------------------
    // 
    // setBTScriptID(...)
    // 
    // Purpose: Set scriptID for BridgeTalk message (SpiderMonkey targets require
    //          an OS dependent format instead of an URI)
    // 
    //-----------------------------------------------------------------------------

    session.setBTScriptID = function( bt, scriptID )
    {
	    bt.headers.ScriptID = this.getBTScriptID( scriptID );
    }

	session.getBTScriptID = function( scriptID )
	{
	    var id		= scriptID;
		var fileOS	= false;

		try
		{
			fileOS = this.cdic.targets[BTBackend.createSpecifier( this.address )].__p__.fileOS;
		}
		catch( exc )
		{}

	    if( fileOS )
	    {
	        //
	        // SpiderMonkey: convert scriptID to os filename
	        //
	        if( typeof id == "string" && ( id[0] == '/' || id[0] == '~') )
	        {
		        var f = File( id );
    		    
		        if( f.exists )
			        id = f.fsName;
	        }
	    }

		return id;
	}

	session.getDbgScriptID = function( scriptID )
	{
	    var id		= scriptID;
		var fileOS	= false;

		try
		{
			fileOS = this.cdic.targets[BTBackend.createSpecifier( this.address )].__p__.fileOS;
		}
		catch( exc )
		{}

	    if( fileOS )
	    {
			if( typeof id == "string" )
			{
				if( id[0] != '~' && id[0] != '(' && id[0] != '{' && id[0] != '[' )
				{
					if( File.fs != "Windows" || ( File.fs == "Windows" && id[0] != '/' ) )
					{
						var f = File( id );
		    		    
						if( f.exists )
							id = f.absoluteURI;
					}
				}
			}
	    }

		return id;
	}

    //-----------------------------------------------------------------------------
    // 
    // processBreakXML(...)
    // 
    // Purpose: Process incoming break info. The passed XML should have the
    //          following format:
    //
    //          <break engine="name" flags="n>
    //              [<error id="n">message</error>]
    //              <stack>
    //                  <frameinfo>text</frameinfo>
    //                  ...more frame info records...
    //              </stack>
    //              <frame engine="name" type="script" file="filename" line="line" name="name">
    //                  <param type="type">value</param>
    //                  ...
    //                  <properties object="">...</properties>
    //                  <source>source</source>
    //              </frame>
    //              <profiling engine="name">
    //                  <file name="filename">
    //                      <function [name="foo"]>
    //                          <data line="n" [time="microseconds"] hits="n"/>
    //                          ...more data elements, one per line...
    //                      </function>
    //                      ...more function elements, one per function...
    //                  </file>
    //                  ...more file elements, one per file...
    //              <profiling>
    //          </break>
    // 
    //-----------------------------------------------------------------------------

    session.processBreakXML = function( xml )
    {	
        if( !this.enabled )
        {
            //
            // The session object is still disabled, therefore defer the call
            // (will be called with next onTick call)
            // A disabled session object was requested by the debugger or
            // the CDIC itself, but wasn't accepted by its counterpart, yet.
            //
            this.pendingXMLCalls.push( xml );
        }
        else
        {
           if( xml.hasOwnProperty( "error" ) )
           {
                //
                // send error message to the debugger
                //
                var errorMsg = xml.error.toString();
                var errorID  = parseInt( xml.error.@id.toString(), 10 );
                
                try
                {
                    this.executionError( errorMsg ).submit();
                }
                catch( exc )
                {}
            }

             if( xml.hasOwnProperty( "profiling" ) )
                 this.profileData = xml.profiling;
            
             if( xml.hasOwnProperty( "frame" ) && xml.frame.@type == "script" && xml.hasOwnProperty( "stack" ) )
             {
                //
                // create a break context which contains all relevant information
                // about the break
                //
                var context = this.createBreakContext( xml.frame, xml.stack );

				if( this.resetLevel )
				{
					//
					// If the flag resetLevel is set, then the script was started
					// using the BT header based protocoll. I.e. continue execution
					// immediately.
					//
					this.resetLevel = false;

					//
					// Only continue if this is not a real breakpoint
					//
					var bp = false;
					
					for( var i=0; i<this.breakpoints.length; i++ )
					{
						if( this.breakpoints[i].line == context.line+1			&& 
							this.breakpoints[i].enabled							&& 
							this.breakpoints[i].scriptID == context.scriptID		)
						{
							bp = true;
							break;
						}
					}
					
					//
					// no breakpoint found, continue
					//
					if( !bp )
					{				            
						this.doContinueXML( this.tmpProfileLevel, true );						
						return;
					}
				}

                try
                {
                    // anounce a execution break
                    //this.executionBreak( context ).submit();
                    sessionmanagerglo.announceSessionDataAvailable(xml);
                }
                catch( exc )
                {}
            }
        }
    }

    //-----------------------------------------------------------------------------
    // 
    // createBreakContext(...)
    // 
    // Purpose: Create BreakContext object and fill with data from passed xml
    // 
    //-----------------------------------------------------------------------------

    session.createBreakContext = function( frameXML, stackXML )
    {
        var context = new BreakContext;
   
        var script  = "";
        var source  = "";
        var line    = -1;
        var frame   = -1;
        var stack   = [];
        
        if( frameXML )
        {
            script  = this.getDbgScriptID( frameXML.@file.toString() );
            source  = frameXML.source.toString();
			line    = isFinite( frameXML.@line.toString() ) ? parseInt( frameXML.@line.toString(), 10 ) : NaN;
			frame   = isFinite( frameXML.@name.toString() ) ? parseInt( frameXML.@name.toString(), 10 ) : NaN;
        }
        
        if( stackXML )
        {
            this.stack  = [];
            
            for each( var stackframe in stackXML )
                this.stack.push( stackframe.toString() );
        }
        
        stack = this.stack;
        
        if( isNaN( line ) )
            line = -1;
        else
            line--;     // debugger expects a 0-based line number
            
        if( isNaN( frame ) )
            frame = this.stack.length - 1;
            
        context.scriptID    = script;
        context.source      = source;
        context.line        = line,
        context.frame       = frame;
        context.stack       = stack;
        
        return context;
    }

    //-----------------------------------------------------------------------------
    // 
    // sendBreakpoints(...)
    // 
    // Purpose: Send breakpoint list to target/engine
    // 
    //-----------------------------------------------------------------------------

    session.sendBreakpoints = function( dontBreakOnErrors )
    {
        if( this.protocol != BTBackend.NOXML )
            this.sendBreakpointsXML( dontBreakOnErrors );
        else
            this.sendBreakpointsNoXML( dontBreakOnErrors );
    }

    session.sendBreakpointsNoXML = function( dontBreakOnErrors )
    {
        function sortsid( a, b )
        {
            if( a.scriptID < b.scriptID )
                return -1;
            if( a.scriptID > b.scriptID )
                return 1;
            return 0;
        }
    
        var bps = this.breakpoints;

        if( bps && bps.length )
        {
            bps.sort( sortsid );
        
            var bt = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "SetBreakpoints" );

            // script ID
            // line,state[,condition]   where state is true,false,temp
            // more BPs
            var body = "";
            var currScriptID = "";
            
            for( var i = 0; i < bps.length; i++ )
            {
                var bp = bps[i];
                
                if( bp.scriptID != currScriptID )
				{
					currScriptID = bp.scriptID;
                    body += this.getBTScriptID( bp.scriptID ) + "\n";
				}
                    
                body += ( bp.line + "," + bp.enabled );
                
                if( bp.condition )
                    body += "," + BridgeTalk.encode( bp.condition );

				body += "\n";
            }

            if (body.length)
            {
                bt.headers.Engine = this.address.engine;
                bt.headers.ScriptID = bps[0].scriptID;
                bt.headers.ClearBP = 1;
                bt.body = body;

                bt.safeSend();
            }
        }
    }

    session.sendBreakpointsXML = function( dontBreakOnErrors )
    {
        var xml = '<breakpoints engine="' + this.address.engine + '" flags="' + ( dontBreakOnErrors ? dontBreakOnErrors : 0 ) + '">';

        for( var i=0; i<this.breakpoints.length; i++ )
        {
            xml += '<breakpoint file="';
            xml += this.getBTScriptID( this.breakpoints[i].scriptID );
            xml += '" line="';
            xml += this.breakpoints[i].line;
            xml += '" enabled="';
            xml += this.breakpoints[i].enabled;
            xml += '" hits="';
            xml += this.breakpoints[i].hits;
            xml += '" count="';
            xml += this.breakpoints[i].hitCount;
            xml += '"><![CDATA[';
            xml += this.breakpoints[i].condition;
            xml += ']]></breakpoint>';
        }            
        
        xml += '</breakpoints>';
        
        var bt = BridgeTalk.create( BTBackend.createSpecifier( this.address ) );
        bt.body = xml;
        bt.safeSend();
    }

    //-----------------------------------------------------------------------------
    // 
    // makeVariableScope(...)
    // 
    // Purpose: Create scope string
    // 
    //-----------------------------------------------------------------------------

    session.makeVariableScope = function( scope, includeBase )
    {
	    var scopeElements = scope.split( '/' );
	    var ret         = '';
    	
	    if( scopeElements.length > 0 )
	    {
	        if( scopeElements[0] == '$.global' )
	            ret += scopeElements[0];
	        else if( scopeElements[0] == '$.local' )
	        {
	            if( scopeElements.length > 1 )
	            {
	                scopeElements.shift();
	                ret += scopeElements[0];
	            }
	        }
    	        
	        for( var i=1; i<scopeElements.length; i++ )
	            ret += "['" + scopeElements[i] + "']";
	    }
    	
	    return ret;
    }

    //-----------------------------------------------------------------------------
    // 
    // postProcessBridgeTalk(...)
    // 
    // Purpose: Process BridgeTalk message if it couldn't be proccessed when at the
    //          moment it was received
    // 
    //-----------------------------------------------------------------------------

    session.postProcessBridgeTalk = function( btData )
    {    
        if( !this.enabled )
        {
            BTBackend.log( "[" + (new Date()).valueOf() + "] postProcessBridgeTalk Error: Session Object Disabled.");
            //
            // The session object is still disabled, therefore defer the call
            // (will be called with next onTick call)
            // A disabled session object was requested by the debugger or
            // the CDIC itself, but wasn't accepted by its counterpart, yet.
            //
            this.pendingBTCalls.push( btData );
        }
        else
        {
			if( btData.headers.Command && btData.headers.Command.length > 0 )
            {
                switch( btData.headers.Command )
                {
                    case "Error":	// runtime error
                    {
                        //
                        // An execution error occured
                        //
                        var error = btData.headers.ErrorMessage.replace( /\\n-/g, ' ' );
            			
                        if( error == 'ENGINE BUSY' )
                        {
                            this.executionError( '$$$/ESToolkit/Messages/EngineBusy' ).submit();
                            break;
                        }
                        else
                        {
                            //
                            // if the debugger is active, this is a halt at a throw
                            //
                            if( btData.headers.ErrorCode == 54 )
                                error = '$$$/ESToolkit/Messages/ExceptionThrown';
                        }
                        
                        //
                        // send error message to the debugger
                        //
                        try
                        {
                            this.executionError( error ).submit();
                        }
                        catch( exc )
                        {}
                    }
                    // run thru

                    case "Break":	// breakpoint hit
                    case "Frame":	// stack frame changed
                    {    	
						if( dbgLog )
						{
							BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- " + btData.headers.Command );
							BTBackend.cdi.writeLog( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- " + btData.headers.Command );
						}

                        var line    = parseInt( btData.headers.CurrentLine, 10 );
                        var frame   = parseInt( btData.headers.Frame, 10 );
                        var script  = this.getDbgScriptID( btData.headers.ScriptID );
                        var source	= '';
                        var stack	= '';
        			    
                        if( btData.headers.Command != 'Frame' )
                        {
                            //
                            // the body contains the stack trace, a newline, and the source
                            //
                            var srcStart	= btData.body.indexOf( '\n\n' );
                            var stack		= btData.body;
            				
                            if( srcStart >= 0 )
                            {
                                stack   = btData.body.substr( 0, srcStart + 1 );
                                source	= btData.body.substr( srcStart + 2 );
                            }
                            
                            this.stack = stack.split( '\n' );                    		        
                        
                            if( this.stack[ this.stack.length-1 ].length == 0 )
                                this.stack.pop();
                        }
                          
                        //
                        // the debugger expects the frame number 0 to be the top level frame
                        //  
                        frame = ( this.stack.length - 1 ) - ( ( isNaN( frame ) || frame < 0 ) ? 0 : frame );
        			    
    			        //
    			        // create a break context which contains all relevant information
    			        // about the break
    			        //
                        var context = new BreakContext( script, source, line, frame, this.stack );
        			    
                        if( this.resetLevel )
                        {
                            //
                            // If the flag resetLevel is set, then the script was started
                            // to run. We got the break here to set breakpoints again and
                            // continue (for targets that create engines on the fly)
                            //
                            this.resetLevel = false;

                            this.sendBreakpoints( this.tmpDontBreak );

                            //
                            // Only continue if this is not a real breakpoint
                            //
                            var bp = false;
                            
                            for( var i=0; i<this.breakpoints.length; i++ )
                            {
                                if( this.breakpoints[i].line == context.line+1	&& 
									this.breakpoints[i].enabled					&& 
									this.breakpoints[i].scriptID == script			)
                                {
                                    bp = true;
                                    break;
                                }
                            }
                            
                            //
                            // no breakpoint found, continue
                            //
                            if( !bp )
                            {				                        
                                this.doContinueNoXML( this.tmpProfileLevel, script, true, this.tmpDontBreak );
                                break;
                            }
                        }

                        try
                        {
                            if( btData.headers.Command == 'Frame' )
                                // announce a execution frame switch
                                this.executionFrameSwitch( context ).submit();
                            else
                                // anounce a execution break
                                this.executionBreak( context ).submit();
                        }
                        catch( exc )
                        {}
                    }
                    break;

                    case 'Exit':
                    {
						if( dbgLog )
						{
							BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- " + btData.headers.Command );
							BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- " + btData.headers.Command ) );
						}

						//
                        // the target exit execution
                        //
                        try
                        {
                            this.exitExecution().submit();
                        }
                        catch( exc )
                        {}
                    }   
                    break;
                }
            }
            else
            {			
                //
                // XML based commands
                //
                var xml     = BTBackend.createXML( btData.body );
                var cmd     = ( xml ? xml.name().toString() : "" );
                var engine  = ( xml && xml.@engine ? xml.@engine.toString() : "" );
    		    
	            if( cmd.length > 0 && engine.length > 0 )
	            {
		            switch( cmd )
		            {
		                case "break":						
						{
							if( dbgLog )
							{
								BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- break" );
								BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- break" ) );
							}
							this.processBreakXML( xml );
						}
	                    break;
		                
		                case "exit":
		                {
							if( dbgLog )
							{
								BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- exit" );
								BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} <- exit" ) );
							}

							//
                            // the target exit execution
                            //
                            try
                            {
                                this.exitExecution().submit();
                            }
                            catch( exc )
                            {}
		                }
		                break;
		            }
		        }
            }
        }
    }
    
    //-----------------------------------------------------------------------------
    // 
    // processBridgeTalk(...)
    // 
    // Purpose: handle incomming BridgeTalk message
    // 
    //-----------------------------------------------------------------------------

    session.processBridgeTalk = function( bt )
    {                
        if( bt && bt.type == 'Debug' )
        {
            //
            // Copy all relevant data from the BridgeTalk message
            // to a JavaScript object to do the processing
            // based on this object.
            // So the processing could be easily be defered if required
            //
		    if( this.protocol == BTBackend.NOXML )
		    {
			    //
			    // SpiderMonkey : convert scriptID to URI
			    //
	            var scriptID = bt.headers.ScriptID;

	            if( typeof scriptID == "string" && scriptID [0] != '(' )
	            {
		            var f = File (scriptID);
    		     
		            if( f.exists )
			            bt.headers.ScriptID = f.absoluteURI;
	            }
		    }

            //
            // put relevant BridgeTalk data into object for
            // later processing
            //
            var btData = { body    : bt.body,
                           headers : { Command      : bt.headers.Command,
                                       ErrorMessage : bt.headers.ErrorMessage,
                                       ErrorCode    : bt.headers.ErrorCode,
                                       CurrentLine  : bt.headers.CurrentLine,
                                       Frame        : bt.headers.Frame,
                                       ScriptID     : bt.headers.ScriptID
                                     }
                         };
                         
            this.postProcessBridgeTalk( btData );
        }
    }

    //-----------------------------------------------------------------------------
    // 
    // getScripts(...)
    // 
    // Purpose: Process Job.GET_SCRIPTS
    //          Get script list for target/engine
    // 
    //-----------------------------------------------------------------------------

    session.getScripts = function( job )
    {    
        if( this.protocol == BTBackend.PUREXML )
            this.getScriptsXML( job );
        else
            this.getScriptsNoXML( job );
    }

    session.getScriptsNoXML = function( job )
    {
        var bt            = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "GetScripts" );
        bt.headers.Engine = this.address.engine;
        bt.job            = job;
		bt.session		  = this;

        bt.onOwnError = function(bt)
        {
			try
			{
				this.job.returnError();
			}
			catch( exc )
			{}
        }

        bt.onOwnResult = function(bt)
        {
            var target = this.target.replace( '#estk', '' );

            // clear all lists
            var infoArray = [];
            var reply = bt.splitBody();

            for( var i=0; i<reply.length; i++ )
		    {
			    // if we have an empty line, this is the separator between scripts and additional info
			    if( reply[i].length == 0 )
				    break;
    				
			    // fill in what we have got
			    var displayName = reply [i][0];
			    var scriptID    = reply [i][1];
			    var status		= reply [i][2];
			    var readOnly	= reply [i][3];
			    if (!status)
				    status = "exec";
			    // backwards compatibility
			    if (status == "active")
				    status = "exec";
    			
			    // do we have both?
			    if( displayName.length && scriptID.length )
			    {
				    // Remove .LNK from a display name in case installers failed
				    if( displayName.substr( displayName.length-4 ) == '.LNK' )
					    displayName = displayName.substr( 0, displayName.length-4 );
				    if( displayName.length <= 0 ) 
				        displayName = ' ';	// ScriptUI might run into breakpoint with 0-length strings
    	
	                info = new ScriptInfo( this.session.getDbgScriptID( scriptID ), displayName, ( status == 'exec' ), ( readOnly == true ) );
    	            			
                    var tmp = File( info.scriptID );
                    
                    if( tmp.exists )
					    info.includes = tmp.parent ? tmp.parent.absoluteURI : "/";
                        
				    infoArray.push( info );
			    }
		    }
    		
		    //
		    // do we have extra info?
		    //
		    if (++i < reply.length && reply[i].length && reply [i][0].length)
		    {
		        //
			    // 1st extra line (V48+): the include path
			    //
			    var includePath = reply [i][0];

			    for( var i=0; i<infoArray.length; i++ )
			        infoArray[i].includes = includePath;
		    }
    		
			try
			{
				this.job.returnResult( infoArray );
			}
			catch( exc )
			{}
	    }
    	
	    if( !bt.safeSend() )
	        job.returnError();
    }

    session.getScriptsXML = function( job )
    {
        var xml		= <get-scripts engine="{this.address.engine}"/>;
        var bt		= BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
        bt.job		= job;
		bt.session	= this;
        
        bt.onOwnResult = function( bt )
        {
            //
            // Expected result:
            //
            // <scripts includepath="path">
            //     <script name="name" id="id" state="state" [readonly="true"]/>
            //      ... more scripts ...
            // </scripts>
            //

            var res         = BTBackend.createXML( bt.body );
            var includes    = '';
            var infoArray   = [];
            
            if( res )
            {
                includes = res.@includepath;
                
                var scriptsXML = res.script;
                
                if( scriptsXML )
                {
                    for( var i=0; i<scriptsXML.length(); i++ )
                    {
                        var info = new ScriptInfo( this.session.getDbgScriptID( scriptsXML[i].@id ),
                                                   scriptsXML[i].@name, 
                                                   scriptsXML[i].@state, 
                                                   ( scriptsXML[i].@readonly == "true" ? true : false ) );
                        infoArray.push( info );
                    }
                }
            }
            else
			{
				try
				{
					this.job.returnError();
				}
				catch( exc )
				{}
			}
            
			try
			{
				this.job.returnResult( infoArray );
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = bt.onOwnTimeout = function( bt )
        {
            BTBackend.returnErrorXML( this.job, bt );
        }
    	
        if( !bt.safeSend() )
            job.returnError();
    }

    //-----------------------------------------------------------------------------
    // 
    // getScriptSource(...)
    // 
    // Purpose: Process Job.GET_SOURCE
    //          Get script source for given script ID
    // 
    //-----------------------------------------------------------------------------

    session.getScriptSource = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.getScriptSourceXML( job );
        else
            this.getScriptSourceNoXML( job );
    }

    session.getScriptSourceNoXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "GetScript" );
        bt.headers.Engine   = this.address.engine;
        bt.job              = job;
        
        this.setBTScriptID( bt, job.scriptInfo.scriptID );

        bt.onOwnResult = function (bt)
        {
			try
			{
				this.job.returnResult( bt.body );
			}
			catch( exc )
			{}
        }

        bt.onOwnError = function (bt)
        {
			try
			{
				this.job.returnError();
			}
			catch( exc )
			{}
        }
    	
	    if( !bt.safeSend() )
	        job.returnError();
    }

    session.getScriptSourceXML = function( job )
    {
		var scriptID = this.getBTScriptID( job.scriptInfo.scriptID );
        var xml = <get-script engine="{this.address.engine}" id="{scriptID}"/>;
        var bt  = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
        bt.job  = job;
        
        bt.onOwnResult = function( bt )
        {
            //
            // Expected result:
            //
            // <script engine="name" id="id>Source code</script>
            //

            var res         = BTBackend.createXML( bt.body );
            var source      = '';
            
            if( res )
                source = res.toString();
                
			try
			{
				this.job.returnResult( source );
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = bt.onOwnTimeout = function( bt )
        {
            BTBackend.returnErrorXML( this.job, bt );
        }
    	
        if( !bt.safeSend() )
            job.returnError();
    }

    //-----------------------------------------------------------------------------
    // 
    // putScriptSource(...)
    // 
    // Purpose: Process Job.PUT_SOURCE
    //          Put script source to given script ID
    // 
    //-----------------------------------------------------------------------------

    session.putScriptSource = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.putScriptSourceXML( job );
        else
            this.putScriptSourceNoXML( job );
    }

    session.putScriptSourceNoXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "PutScripts" );
        bt.headers.Engine   = this.address.engine;
        bt.body             = job.source;
        bt.job              = job;
        
        this.setBTScriptID( bt, job.scriptInfo.scriptID );

        bt.onOwnResult = function (bt)
        {
			try
			{
				this.job.returnResult( true );
			}
			catch( exc )
			{}
        }

        bt.onOwnError = function (bt)
        {
			try
			{
				this.job.returnResult( false );
			}
			catch( exc )
			{}
        }
    	
	    if( !bt.safeSend() )
	        job.returnError();
    }

    session.putScriptSourceXML = function( job )
    {
        session.putScriptSourceNoXML( job );
    // TODO
    }

    //-----------------------------------------------------------------------------
    // 
    // eval(...)
    // 
    // Purpose: Process Job.EVAL
    //          Eval script
    // 
    //-----------------------------------------------------------------------------

    session.eval = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.evalXML( job );
        else
            this.evalNoXML( job );
    }

    session.evalNoXML = function( job )
    {
        var source = job.source;
        
        if( source.length )
        {
            var bt                  = BridgeTalk.create( BTBackend.createSpecifier( this.address ), 'Eval' );
            bt.headers.Profiling    = 0;
            bt.headers.Engine       = this.address.engine;
			bt.headers.IgnoreErrors = true;
			bt.headers.DebugFlags   = 1024;
            bt.body                 = source;
            bt.job                  = job;
        	
            bt.onOwnResult = function( bt )
            {
	            // the reply is datatype,result
	            var reply = bt.splitBody();
	            var result = '';
    			
	            if( reply && reply[0] )
	                result = reply[0][1];
    	            
				try
				{
					this.job.returnResult( result );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = function(bt)
            {
				try
				{
					if( bt.body.length )
						this.job.returnError( bt.body );
					else
						this.job.returnError();
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnTimeout = function(bt)
            {
                // ignore timeout
				try
				{
					this.job.quitTask();
				}
				catch( exc )
				{}
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }
        else
            job.returnResult( '' );
    }

    session.evalXML = function( job )
    {
        var source = job.source;
        
        if( source.length )
        {
			var cdataSrc = new XML("<![CDATA[" + source + "]]>");
			var xml = <eval engine="{this.address.engine}" debug="0" profiling="0" flags="1024" ignore-errors="true" shutdown="false"><source>{cdataSrc}</source></eval>;
         
            var bt      = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
            bt.job      = job;
            bt.session  = this;
            
            bt.onOwnResult = bt.onOwnError = function( bt )
            {
                var value   = '';
                var error   = null;
                var errCode = NaN;
                
                try
                {
                    var res = new XML( bt.body );
                    
                    if( res )
                    {
                        value   = res.value.toString();
                        error   = res.error.toString();
                        errCode = parseInt( res.error.@id, 10 );
                        
                        this.session.profileData = res.profiling;
                    }
                }
                catch( exc )
                {
                    value = '';
                }
          
				try
				{
					if( error && error.length )
						this.job.returnError( errCode, error );
					else    
						this.job.returnResult( value );
				}
				catch( exc )
				{}
            }
            
            bt.onOwnTimeout = function( bt )
            {
				try
				{
					// ignore timeout
					this.job.returnError( localize( "$$$/CT/ExtendScript/Errors/Err35=Timeout" ) );
				}
				catch( exc )
				{}
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }
        else
            job.returnResult( ' ' );
    }

    //-----------------------------------------------------------------------------
    // 
    // switchFrame(...)
    // 
    // Purpose: Process Job.SWITCH_FRAME
    //          Switch current frame
    // 
    //-----------------------------------------------------------------------------

    session.switchFrame = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.switchFrameXML( job );
        else
            this.switchFrameNoXML( job );
    }

    session.switchFrameNoXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), 'SwitchFrame' );
        bt.headers.Engine   = this.address.engine;
        bt.body             = this.stack.length - 1 - job.frame;
        bt.job              = job;
        
        bt.onOwnResult = function( bt )
        {   
			try
			{
				this.job.returnResult( true );
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = function( bt )
        {
			try
			{
				this.job.returnResult( false );
			}
			catch( exc )
			{}
        }
        
        if( !bt.safeSend() )
            job.returnResult( false );
    }

    session.switchFrameXML = function( job )
    {
        var frame = this.stack.length - 1 - job.frame;
        var xml   = <set-frame engine="{this.address.engine}" frame="{frame}"/>;
        
        var bt      = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
        bt.job      = job;
        bt.session  = this;
        bt.frame    = job.frame;
        
        bt.onOwnResult = function( bt )
        {
            var xml = BTBackend.createXML( bt.body );
            
            if( xml )
            {
                var context   = this.session.createBreakContext( xml );
                context.frame = this.frame;
                
                try
                {
                    // anounce a execution break

                    //this.session.executionFrameSwitch( context ).submit();
                    sessionmanagerglo.announceSessionDataAvailable(xml);
                }
                catch( exc )
                {}

				try
				{
					this.job.returnResult( true );
				}
				catch( exc )
				{}
            }
            else
			{
				try
				{
					this.job.returnError();
				}
				catch( exc )
				{}
			}
        }
        
        bt.onOwnError = bt.onOwnTimeout = function( bt )
        {
            BTBackend.returnErrorXML( this.job, bt );
        }

        if( !bt.safeSend() )
            job.returnError();
    }

    //-----------------------------------------------------------------------------
    // 
    // getFrame(...)
    // 
    // Purpose: Process Job.GET_FRAME
    //          Get the frame
    // 
    //-----------------------------------------------------------------------------

    session.getFrame = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.getFrameXML( job );
        else
            this.getFrameNoXML( job );
    }

    session.getFrameNoXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), 'GetFrame' );
        bt.headers.Engine   = this.address.engine;
        bt.body             = this.stack.length - 1 - job.frame;
        bt.job              = job;
        
        bt.onOwnResult = function( bt )
        {   
            try
            {
                this.job.returnResult( true );
            }
            catch( exc )
            {}
        }
        
        bt.onOwnError = function( bt )
        {
            try
            {
                this.job.returnResult( false );
            }
            catch( exc )
            {}
        }
        
        if( !bt.safeSend() )
            job.returnResult( false );
    }

    session.getFrameXML = function( job )
    {
        var frame = this.stack.length - 1 - job.frame;
        var xml   = <get-frame engine="{this.address.engine}" frame="{frame}"/>;
        
        var bt      = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
        bt.job      = job;
        bt.session  = this;
        bt.frame    = job.frame;
        
        bt.onOwnResult = function( bt )
        {
            var xml = BTBackend.createXML( bt.body );
            
            if( xml )
            {
                var context   = this.session.createBreakContext( xml );
                context.frame = this.frame;
                
                try
                {
                    // anounce a execution break

                    //this.session.executionFrameSwitch( context ).submit();
                    sessionmanagerglo.announceSessionDataAvailable(xml);
                }
                catch( exc )
                {}

                try
                {
                    this.job.returnResult( true );
                }
                catch( exc )
                {}
            }
            else
            {
                try
                {
                    this.job.returnError();
                }
                catch( exc )
                {}
            }
        }
        
        bt.onOwnError = bt.onOwnTimeout = function( bt )
        {
            BTBackend.returnErrorXML( this.job, bt );
        }

        if( !bt.safeSend() )
            job.returnError();
    }


    //-----------------------------------------------------------------------------
    // 
    // startExecution(...)
    // 
    // Purpose: Process Job.START_EXECUTION
    //          Start debugging session
    // 
    //-----------------------------------------------------------------------------

    session.startExecution = function( job )
    {   
        if( !job.context )
            job.returnError();
        else
        {
            var specifier = BTBackend.createSpecifier( this.address );
/*  
	pwollek 05/14/2008
	Hotfix for #1785159 where ID always returned a status of 'BUSY' so that a script
	never got executed.
	
            const timespan  = 5000;         // 5 sec. timeout
            var startStatus = new Date();
            var endStatus   = startStatus;
            var status      = "BUSY";
            
            do
            {
                status      = BridgeTalk.getStatus( specifier );
                endStatus   = new Date();
                
            } while( status == "BUSY" && ( endStatus - startStatus ) < timespan )
            
            if( status != "BUSY" )
*/			
            {	   
				if( this.features[Feature.TO_FRONT].supported && specifier != BridgeTalk.appSpecifier )
					BridgeTalk.bringToFront( specifier );
                
                this.breakpoints    = [];
                
                for( var i=0; i<job.context.breakpoints.length; i++ )
                    this.breakpoints.push( job.context.breakpoints[i] );

                if( this.protocol == BTBackend.PUREXML )
                    this.startExecutionXML( job );
                else
                    this.startExecutionNoXML( job );
            }
/*
	pwollek 05/14/2008: see comment above
	
            else
            {
                var msg = localize( "$$$/ESToolkit/Error/EngineBusy=Target engine '%1' is busy!",
			                        this.address.engine );
			    
			    job.returnError( msg );
            }
*/			
        }
    }

    session.startExecutionNoXML = function( job )
    {    
        this.dontBreakFlag    = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );

        //
        // If the debug level is 1 (call debugger on break) then set
        // debug level to 2 (call debugger immediately) to set breakpoints
        // again (for targets that create engines on the fly)
        //
        if( job.context.debugLevel == ExecutionContext.DBGLEVEL_BREAK )
        {
            job.context.debugLevel = ExecutionContext.DBGLEVEL_BREAKIMMIDEATE;
            this.resetLevel         = true;
            this.tmpDontBreak       = this.dontBreakFlag;
            this.tmpProfileLevel    = job.context.profileLevel;
        }

        var specifier = BTBackend.createSpecifier( this.address );
        
        var bt                  = BridgeTalk.create( specifier, 'Eval' );
        bt.headers.Engine       = this.address.engine;
        bt.headers.DebugLevel   = job.context.debugLevel;
        bt.headers.Profiling    = job.context.profileLevel;
        bt.headers.DebugFlags   = this.dontBreakFlag;
        bt.session              = this;
        bt.job                  = job;
        bt.body                 = "//";
		
		if( job.context.source && job.context.source.length )
			bt.body = job.context.source;
        
        this.setBTScriptID( bt, job.context.scriptID );
    	
        bt.onOwnResult = function( bt )
        {
            // the reply is datatype,result
            var reply = bt.splitBody();
    		
			try
			{
				if( reply && reply.length && reply[0].length )
					this.job.returnResult( reply[0][1] );
				else
					this.job.quitTask();
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = function (bt)
        {
            //
            // display the error message
            // if there is no line number and script name info, use an alert
            // but not if the error is "Execution Halted"
            //
            var errCode = parseInt( bt.headers[ 'Error-Code' ], 10 );
            var msg = bt.body;
        
            if( !bt.headers.ScriptID && errCode != -34 )
            {
                //
                // if the error code is 32 (kErrBadAction), assume that there
                // was no engine to execute in.
                //
                
                if( errCode == 32 )
                {
                    msg = localize ("$$$/ESToolkit/Error/MissingEngine=Cannot execute script in target engine '%1'!",
                                    this.headers.Engine);
                }
                else if( msg == "ENGINE BUSY" )
                    msg = localize ("$$$/ESToolkit/Error/EngineBusy=Target engine '%1' is busy!",
                                    this.headers.Engine);
            }
            
			try
			{
				this.job.returnError( errCode, msg );
			}
			catch( exc )
			{}
        }
    	
        if( !bt.safeSend() )
            bt.onOwnError(bt);
		else if( dbgLog )
		{
			BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> StartExecution" );
			BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> StartExecution" ) );
		}

    }

    session.startExecutionXML = function( job )
    {	
		this.dontBreakFlag    = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );

		var source = "//";
		
		if( job.context.source && job.context.source.length )
			source = job.context.source;

		var scriptID = this.getBTScriptID( job.context.scriptID );
		var cdataSrc = new XML("<![CDATA[" + source + "]]>");
		var xml = <eval engine="{this.address.engine}" file="{scriptID}" debug="{job.context.debugLevel}" profiling="{job.context.profileLevel}" all="true" flags="{this.dontBreakFlag}" ><source>{cdataSrc}</source></eval>;
        var bps = <breakpoints/>;
        
        for( var b=0; b<this.breakpoints.length; b++ )
		{
			var scriptID = this.getBTScriptID( this.breakpoints[b].scriptID );
			var cdataCond = new XML("<![CDATA[" + this.breakpoints[b].condition + "]]>");
            bps.appendChild( <breakpoint file="{scriptID}" line="{this.breakpoints[b].line}" hits="{this.breakpoints[b].hits}" count="{this.breakpoints[b].hitCount}" enabled="{this.breakpoints[b].enabled}">{cdataCond}</breakpoint> );
		}

		xml.appendChild( bps );

        var specifier   = BTBackend.createSpecifier( this.address );
        var bt          = BridgeTalk.createXML( specifier, xml, true );
        bt.job          = job;
        bt.session      = this;

		bt.timeout = 604800;	// a week

        bt.onOwnResult = function( bt )
        {
            var value   = '';
            var error   = null;
            var errCode = NaN;
            try
            {
                var res = new XML( bt.body );
                
                if( res )
                {
                    value   = res.value.toString();
                    error   = res.error.toString();
                    errCode = parseInt( res.error.@id, 10 );
                    
                    this.session.profileData = res.profiling;
                }
            }
            catch( exc )
            {
                value = '';
            }
      
			try
			{
				if( error && error.length )
					this.job.returnError( errCode, error );
				else    
					this.job.returnResult( value );
			}
			catch( exc )
			{}
        }

        bt.onOwnError = function( bt )
        {
			try
			{
				BTBackend.returnErrorXML( this.job, bt );
			}
			catch( exc )
			{}
        }   

        if( !bt.safeSend() )
            job.returnError();
		else if( dbgLog )
		{
			BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> startExecution" );
			BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> startExecution" ) );
		}
    }

    //-----------------------------------------------------------------------------
    // 
    // continueExecution(...)
    // 
    // Purpose: Process Job.CONTINUE_EXECUTION
    //          Continue debugging session
    // 
    //-----------------------------------------------------------------------------

    session.continueExecution = function( job )
    {
        if( !job.context )
            job.returnError();
        else
        {
            if( this.protocol == BTBackend.PUREXML )
                this.continueExecutionXML( job );
            else
                this.continueExecutionNoXML( job );
        }
    }

    session.continueExecutionNoXML = function( job )
    {
        this.dontBreakFlag  = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );
        var ignoreErrors	= ( job.context.flags & ExecutionContext.DBGFLAG_IGNOREERROR ? 1 : 0 );
        
        var ret = this.doContinueNoXML( job.context.profileLevel, job.context.scriptID, ignoreErrors, this.dontBreakFlag );
    	
		try
		{
			job.returnResult( ret );
		}
		catch( exc )
		{}
    }

    session.continueExecutionXML = function( job )
    {
        this.dontBreakFlag  = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );
        var ignoreErrors	= ( job.context.flags & ExecutionContext.DBGFLAG_IGNOREERROR ? true : false );

		try
		{
			job.returnResult( this.doContinueXML( job.context.profileLevel, ignoreErrors ) );
		}
		catch( exc )
		{}
    }
	
	session.doContinueXML = function( profileLevel, ignoreErrors )
	{
		var xml = <continue engine="{this.address.engine}" profiling="{profileLevel}" flags="{this.dontBreakFlag}" ignore-errors="{ignoreErrors}" shutdown="false"/>;

		if( xml )
		{
			var specifier = BTBackend.createSpecifier( this.address );
			
			if( this.features[Feature.TO_FRONT].supported && specifier != BridgeTalk.appSpecifier )
				BridgeTalk.bringToFront( specifier );
			
			var bt     = BridgeTalk.createXML( specifier, xml );
			bt.session = this;
			
			bt.onOwnError = function( bt )
			{
				var errMsg  = "";
				
				try
				{
					var xml = new XML( bt.body );
					
					if( xml )
						errMsg  = xml.toString();
				}
				catch( exc )
				{}
				
				if( errMsg.length > 0 )
					this.session.executionError( errMsg ).submit();
			}
			
			if( dbgLog )
			{
                BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> continueExecution" );
				BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> continueExecution" ) );
			}

			return bt.safeSend();
		}
		else
			return false;
	}

    session.doContinueNoXML = function( profileLevel, scriptID, ignoreErrors, dontBreak )
    {
        this.dontBreakFlag      = dontBreak;
        
        var specifier = BTBackend.createSpecifier( this.address );
        
		if( this.features[Feature.TO_FRONT].supported && specifier != BridgeTalk.appSpecifier )
			BridgeTalk.bringToFront( specifier );
        
        var bt                  = BridgeTalk.create( specifier, 'Continue' );
        bt.headers.Engine       = this.address.engine;
        bt.headers.Profiling    = profileLevel;
        bt.headers.IgnoreErrors = ignoreErrors;
        bt.headers.DebugFlags   = this.dontBreakFlag;
        
        this.setBTScriptID( bt, scriptID );

        // re-attach existing breakpoints
        this.sendBreakpoints( this.dontBreakFlag );
        
		if( dbgLog )
		{
			BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> ContinueExecution" );
			BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> ContinueExecution" ) );
		}

        return bt.safeSend();
    }
    
    //-----------------------------------------------------------------------------
    // 
    // debuggingCommand(...)
    // 
    // Purpose: 
    // 
    //-----------------------------------------------------------------------------

    session.debuggingCommand = function( job, cmd )
    {
        if( !job.context )
            job.returnError();
        else if( cmd == BTBackend.CmdStop       || 
                 cmd == BTBackend.CmdPause      || 
                 cmd == BTBackend.CmdStepOver   || 
                 cmd == BTBackend.CmdStepInto   || 
                 cmd == BTBackend.CmdStepOut        )
        {
            if( this.protocol == BTBackend.PUREXML )
                this.debuggingCommandXML( job, cmd );
            else
                this.debuggingCommandNoXML( job, cmd );
        }
    }
    
    session.debuggingCommandNoXML = function( job, cmd )
    {
        this.dontBreakFlag  = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );
        var ignoreErrors	= ( job.context.flags & ExecutionContext.DBGFLAG_IGNOREERROR ? 1 : 0 );

        // re-attach existing breakpoints
        if( cmd != BTBackend.CmdStop && cmd != BTBackend.CmdPause )
            this.sendBreakpoints( this.dontBreakFlag );
    	
    	var specifier = BTBackend.createSpecifier( this.address );
    	
        var bt                  = BridgeTalk.create( specifier, cmd[0] );
        bt.headers.Engine       = this.address.engine;
        bt.headers.Profiling    = job.context.profileLevel;
        bt.headers.IgnoreErrors = ignoreErrors;
        bt.headers.DebugFlags   = this.dontBreakFlag;
        bt.job                  = job;
        
        this.setBTScriptID( bt, job.context.scriptID );
	    
        bt.onOwnResult = function( bt )
        {   
			try
			{
				this.job.returnResult( true );
			}
			catch( exc )
			{}
        }
	    
        bt.onOwnError = function( bt )
        {
			try
			{
				this.job.returnResult( false );
			}
			catch( exc )
			{}
        }
	    
        if( !bt.safeSend() )
            job.returnResult( false );
		else if( dbgLog )
		{
			BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> " + cmd[0] );
			BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> " + cmd[0] ) );
		}
    }
    
    session.debuggingCommandXML = function( job, cmd )
    {
        this.dontBreakFlag  = ( job.context.flags & ExecutionContext.DBGFLAG_DONTBREAKONERROR ? 1024 : 0 );
        var ignoreErrors	= ( job.context.flags & ExecutionContext.DBGFLAG_IGNOREERROR ? true : false );

        var xml = <{cmd[1]} engine="{this.address.engine}" profiling="{job.context.profileLevel}" flags="{this.dontBreakFlag}" ignore-errors="{ignoreErrors}" shutdown="false"/>;

        if( xml )
        {
	        var specifier = BTBackend.createSpecifier( this.address );
        	
            var bt     = BridgeTalk.createXML( specifier, xml );
            bt.session = this;
            
            bt.onOwnError = function( bt )
            {
                var errMsg  = "";
                
                try
                {
                    var xml = new XML( bt.body );
                    
                    if( xml )
                        errMsg  = xml.toString();
                }
                catch( exc )
                {}
                
                if( errMsg.length > 0 )
                    this.session.executionError( errMsg ).submit();
            }
            
			if( dbgLog )
			{
				BTBackend.log( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> " + cmd[1] );
				BTBackend.cdi.writeLog( ( "[" + (new Date()).valueOf() + "] {" + this.address + "} -> " + cmd[1] ) );
			}

            job.returnResult( bt.safeSend() );
        }
        else
            job.returnError();
    }

    //-----------------------------------------------------------------------------
    // 
    // reset(...)
    // 
    // Purpose: Process Job.RESET
    //          Reset the engine
    // 
    //-----------------------------------------------------------------------------

    session.reset = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
        {
            var xml = '<eval engine="' + this.address.engine + '" reset="true"/>';
            var bt = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml );
            
            job.returnResult( bt.safeSend() );
        }
        else
            job.returnError();
    }
    
    //-----------------------------------------------------------------------------
    // 
    // getVariables(...)
    // 
    // Purpose: Process Job.GET_VARIABLES
    //          Get variables of engine
    // 
    //-----------------------------------------------------------------------------

    session.getVariables = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.getVariablesXML( job );
        else
            this.getVariablesNoXML( job );
    }

    session.getVariablesNoXML = function( job )
    {
	    // create the setting for things to exclude
	    var excludes = [];
    	
	    if( job.excludes & Variable.TYPE_UNDEFINED )   excludes.push ("undefined");
	    if( job.excludes & Variable.TYPE_CORE)         excludes.push ("builtin");
	    if( job.excludes & Variable.TYPE_FUNCTION )    excludes.push ("Function");
	    if( job.excludes & Variable.TYPE_PROTOTYPE )   excludes.push ("prototype");
    	
	    var bt                      = BridgeTalk.create( BTBackend.createSpecifier( this.address ), 'Variables' );
	    bt.headers.Engine           = this.address.engine;
	    bt.headers.Exclude          = excludes.join( ',' );
	    bt.headers.MaxArrayElements = job.maxArrayElements;
	    bt.headers.All              = "1";
	    bt.job                      = job;
	    bt.body                     = this.makeVariableScope( job.scope );
		bt.session					= this;

	    bt.onOwnResult = function (bt)
	    {
			/*
			This handler for the old API converts the old format into the new XML format.
			*/
	        var lines = bt.body.split ('\n');

			var xml = <properties engine={this.headers.Engine} object={this.body}/>;

	        for( var i=0; i<lines.length; i++ )
	        {
	            //
		        // now split the line manually to digest escaped characters.
		        //
		        var line    = [];
		        var s       = lines [i];
		        var cur     = '';
    		    
		        for( var j=0; j<s.length; j++ )
		        {
			        switch( s[j] )
			        {
				        case '\\': cur += (s [++j] == 'n') ? '\n' : s[j]; break;
				        case ',':  line.push (cur); cur = ""; break;
				        default:   cur += s[j];
			        }
		        }
		        if( cur.length )
			        line.push( cur );
		        if( line.length < 3 )
			        continue;
    			
				var dataType = line [0];
				var name     = line [1];
				var value    = line [2];
				var readonly = (line [3] == 'true');
				var valid    = (dataType.indexOf ('invalid' ) < 0);
		        // If the line is an invalid object, the data type is "invalid classname"
		        if( !valid )
			        dataType = dataType.substr(8);
				if (dataType == "EnumError")
					dataType = "error";

				var prop = <property name={name} type={dataType}>{escape(value)}</property>;
				if (readonly)
					prop.@readonly = "true";
				if (!valid)
					prop.@invalid = "true";
				xml.appendChild (prop);
	        }

			try
			{
				this.job.returnResult( xml );
			}
			catch( exc )
			{}
	    }
    	
	    bt.onOwnError = function(bt)
	    {
			try
			{
				this.job.returnError();
			}
			catch( exc )
			{}
	    }
    	
        if( !bt.safeSend() )
            job.returnError();
    }

    session.getVariablesXML = function( job )
    {
        var _excludes_ = [];

        if( job.excludes & Variable.TYPE_UNDEFINED )   _excludes_.push ("undefined");
        if( job.excludes & Variable.TYPE_CORE)         _excludes_.push ("builtin");
        if( job.excludes & Variable.TYPE_FUNCTION )    _excludes_.push ("Function");
        if( job.excludes & Variable.TYPE_PROTOTYPE )   _excludes_.push ("prototype");

        var excludes    = _excludes_.join( ',' );
        var scope       = job.scope;
        var max         = job.maxArrayElements;
        
        var xml = <get-properties engine="{this.address.engine}" object="{scope}" exclude="{excludes}" all="true" max="{max}"/>;
        var bt  = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml );
        bt.job  = job;

        bt.onOwnResult = function( bt )
        {
			try
			{
                sessionmanagerglo.announceSessionDataAvailable(bt.body);
				this.job.returnResult( BTBackend.createXML( bt.body ) );
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = onOwnTimeout = function( bt )
        {
			try
			{
                sessionmanagerglo.announceSessionDataAvailable(bt.body);
				BTBackend.returnErrorXML( this.job, bt );
			}
			catch( exc )
			{}
        }
        
        if( !bt.safeSend() )
            job.returnError();
    }

    //-----------------------------------------------------------------------------
    // 
    // getVariableValue(...)
    // 
    // Purpose: Process Job.GET_VARIABLE_VALUE
    //          Get value of a single variable
    // 
    //-----------------------------------------------------------------------------

    session.getVariableValue = function( job )
    {
		this.getVariableValueNoXML( job );
    }

    session.getVariableValueNoXML = function( job )
    {
        var scope = job.scope;
        
        if( scope.length > 0 )
            scope = this.makeVariableScope( job.scope );
        
	    var bt               = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "Eval" );
	    bt.headers.Engine    = this.address.engine;
	    bt.headers.WalkStack = 1;   // we want the target to walk the stack until we have a result
	    bt.varName           = job.variable;
	    bt.job               = job;

	    bt.onOwnResult = function (bt) 
	    {
	        var variable = null;

		    // the reply of eval is datatype,result
		    var reply    = bt.splitBody();
		    var dataType = '';
		    var value    = '';
    		
		    if( reply && reply.length && reply[0].length )
		    {
		        dataType = reply [0][0];
		        value    = reply [0][1];
    		    
		        variable = new Variable( this.varName );

		        switch( dataType )
		        {
			        case "undefined":	
				        variable.type  = Variable.TYPE_UNDEFINED;
				        variable.value = undefined;
				        break;

			        case "null":		
				        variable.type  = Variable.TYPE_NULL;
				        variable.value = null;
				        break;

			        case "boolean":		
				        variable.type  = Variable.TYPE_BOOLEAN;
				        variable.value = ( value == 'true' );
				        break;

			        case "number":		
				        variable.type  = Variable.TYPE_NUMBER;
				        variable.value = Number( value );
				        break;

			        case "string":		
				        variable.type  = Variable.TYPE_STRING;
				        variable.value = ( typeof value == 'undefined' ? '' : value );
				        break;

			        case "Function":
				        // a function may be defined in several ways:
				        // Function,foo,foo()   defined as function foo(){}
				        // Function,foo,()      defined as foo = function()
				        // Function,bar,foo()   defined as bar = foo
				        variable.type  = Variable.TYPE_FUNCTION;
				        variable.value = ( value[0] == '(' ? value = variable.name + value : value );
				        break;

			        default:
				        // show the class if the value does not start with '['
				        variable.type  = Variable.TYPE_OBJECT;

				        if( typeof value == "undefined")
					        variable.value = '[' + dataType + ']';
				        else if( value.length && value[0] != '[' )
					        variable.value = '[' + dataType + '] ' + value;
		        }
		    }
    		
			try
			{
				this.job.returnResult( variable );
			}
			catch( exc )
			{}
	    }

	    // the error handler displays the error
	    bt.onOwnError = function (bt)
	    {
			try
			{
				this.job.quitTask();
			}
			catch( exc )
			{}
	    }
    	
	    bt.body = job.variable;
    	
	    if( !bt.safeSend() )
	        job.quitTask();
    }

    //-----------------------------------------------------------------------------
    // 
    // setVariableValue(...)
    // 
    // Purpose: Process Job.SET_VARIABLE_VALUE
    //          Set value of variable
    // 
    //-----------------------------------------------------------------------------

    session.setVariableValue = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.setVariableValueXML( job );
        else
            this.setVariableValueNoXML( job );
    }

    session.setVariableValueNoXML = function( job )
    {
        var source = this.makeVariableScope( job.scope ) + '=' + unescape(job.value);
        
        if( source.length )
        {
            var bt                  = BridgeTalk.create( BTBackend.createSpecifier( this.address ), 'Eval' );
            bt.headers.Profiling    = 0;
            bt.headers.Engine       = this.address.engine;
            bt.body                 = source;
            bt.job                  = job;
        	
            bt.onOwnResult = function( bt )
            {
				try
				{
					this.job.returnResult( true );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = function(bt)
            {
				try
				{
					this.job.returnResult( false );
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnTimeout = function(bt)
            {
				try
				{
					this.job.returnResult( false );
				}
				catch( exc )
				{}
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }
        else
            job.returnResult( false );
    }

    session.setVariableValueXML = function( job )
    {
        var source = job.scope + '=' + unescape(job.value);
        
        if( source.length )
        {
			var cdataSrc = new XML("<![CDATA[" + source + "]]>");
            var xml = <eval engine="{this.address.engine}" debug="0" profiling="0"><source>{cdataSrc}</source></eval>;
            
            var bt = BridgeTalk.createXML( BTBackend.createSpecifier( this.address ), xml, true );
            bt.job = job;
                        	
            bt.onOwnResult = function( bt )
            {
				var error = null;

                try
                {
                    var res = new XML( bt.body );
                    
                    if( res )
                        error = res.error.toString();
				}
				catch( exc )
				{}

				try
				{
					if( error && error.length )
					{
                        var err = "<set-result><error>" + error + "</error></set-result>";
                        sessionmanagerglo.announceSessionDataAvailable(err);
						this.job.returnError();
					}
                    else
                    {
                        var res = "<set-result>true</set-result>";
                        sessionmanagerglo.announceSessionDataAvailable(res);
						this.job.returnResult( true );
                    }
				}
				catch( exc )
				{}
            }
        	
            bt.onOwnError = bt.onOwnTimeout = function(bt)
            {
				try
				{
					this.job.returnError();
				}
				catch( exc )
				{}
            }
        	
            if( !bt.safeSend() )
                job.returnError();
        }
    }

    //-----------------------------------------------------------------------------
    // 
    // getProfileData(...)
    // 
    // Purpose: Process Job.GET_PROFILE_DATA
    //          get current profile data
    // 
    //-----------------------------------------------------------------------------

    session.getProfileData = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.getProfileDataXML( job );
        else
            this.getProfileDataNoXML( job );
    }

    session.getProfileDataNoXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "ProfilerData" );
        bt.headers.Engine   = this.address.engine;
        bt.job              = job;
        
    //	    if (erase)
    //		    bt.headers.Clear = "1";
		/*  
		<profiling>
			<file name="filename">
				<function name="foo">
					<data line="n" time="usecs" hits="n"/>
				</function>
			</file>
		<profiling>
		*/
        bt.onOwnResult = function(bt)
        {
			var xml		= <profiling/>;
			var fileXML = <file/>;
			var funcXML = <function/>;

	        var reply       = bt.splitBody();
	        var allScriptID = null;
			var fileName	= null;
			var funcName	= null;

			var allFiles = {};		// indexed by scriptID
			var allFunctions = {};	// indexed by scriptID and function name

			for( var i=0; i<reply.length; i++ )
	        {	
				var rec = reply [i];
    	        
				if (rec.length == 5 && rec[4] !== fileName)
				{
					fileName = rec[4];
					fileXML = allFiles [fileName];
					if (!fileXML)
					{
						fileXML = <file name={fileName}/>;
						allFiles [fileName] = fileXML;
						xml.appendChild (fileXML);
					}
					funcName = null;
				}
				if (rec.length >= 4 && rec[3] !== funcName)
				{
					funcName = rec[3];
					funcXML = allFunctions [fileName + '/' + funcName];
					if (!funcXML)
					{
						funcXML = <function name={funcName}/>;
						allFunctions [fileName + '/' + funcName] = funcXML;
						fileXML.appendChild (funcXML);
					}
				}
				funcXML.appendChild (<data line={+rec[0]+1} time={rec[1]} hits={rec[2]}/>);
			}
    	    
			try
			{
				this.job.returnResult( xml );
			}
			catch( exc )
			{}
        }
        
        bt.onOwnError = function(bt)
        {
			try
			{
				this.job.quitTask();
			}
			catch( exc )
			{}
        }
        
        if( !bt.safeSend() )
            job.quitTask();
    }

    session.getProfileDataXML = function( job )
    {
        job.returnResult( this.profileData );
    }

    //-----------------------------------------------------------------------------
    // 
    // sendBreakpoints(...)
    // 
    // Purpose: Process Job.SET_BREAKPOINTS
    //          Send breakpoints to target
    // 
    //-----------------------------------------------------------------------------

    session.updateBreakpoints = function( job )
    {    
        this.breakpoints    = [];
        
        for( var i=0; i<job.breakpoints.length; i++ )
            this.breakpoints.push( job.breakpoints[i] );
            
        this.sendBreakpoints( this.dontBreakFlag );
    }

    //-----------------------------------------------------------------------------
    // 
    // removeBreakpoints(...)
    // 
    // Purpose: Process Job.REMOVE_BREAKPOINTS
    //          Remove breakpoints
    // 
    //-----------------------------------------------------------------------------

    session.removeBreakpoints = function( job )
    {    
        this.breakpoints    = [];
        this.sendBreakpoints( this.dontBreakFlag );
    }

    //-----------------------------------------------------------------------------
    // 
    // getBreakpoints(...)
    // 
    // Purpose: Process Job.GET_BREAKPOINTS
    //          Get breakpoints from engine
    // 
    //-----------------------------------------------------------------------------

    session.getBreakpoints = function( job )
    {
        if( this.protocol == BTBackend.PUREXML )
            this.getBreakpointsXML( job );
        else
            this.getBreakpointsNoXML( job );
    }

    session.getBreakpointsNoXML = function( job )
    {
    // TODO
    }

    session.getBreakpointsXML = function( job )
    {
        var bt              = BridgeTalk.create( BTBackend.createSpecifier( this.address ), "" );
        bt.body             = '<get-breakpoints engine="' + this.address.engine + '"/>';
        bt.job              = job;

        bt.onOwnResult = function (bt)
        {
            var bps = [];
            
            if( bt.body.length > 0 )
            {
                var xml = BTBackend.createXML( bt.body );
                
                if( xml )
                {
					xml = xml.breakpoint;

					if( xml )
					{
						for( var i=0; i<xml.length(); i++ )
						{
							var bp = new Breakpoint( parseInt( xml[i].@line, 10 ), xml[i].@file, ( xml[i].@enabled.toString() != 'false' ), parseInt( xml[i].@hits, 10 ), parseInt( xml[i].@count, 10 ), xml[i].children(0) );
							bps.push( bp );
						}
					}
                }
            }
            
            try
			{
				this.job.returnResult( bps );
			}
			catch( exc )
			{}
        }

        bt.onOwnError = function (bt)
        {
			try
			{
				this.job.quitTask();
			}
			catch( exc )
			{}
        }
    	
	    if( !bt.safeSend() )
	        job.quitTask();
    }

    //-----------------------------------------------------------------------------
    // 
    // onTask(...)
    // 
    // Purpose: Dispatch CDI tasks
    // 
    //-----------------------------------------------------------------------------

    session.onTask = function( job )
    {
        switch( job.name )
        {
            case Job.START_EXECUTION:
                this.startExecution( job );
                break;

            case Job.CONTINUE_EXECUTION:
                this.continueExecution( job );
                break;

            case Job.STOP_EXECUTION:
                this.debuggingCommand( job, BTBackend.CmdStop );
                break;

            case Job.PAUSE_EXECUTION:
                this.debuggingCommand( job, BTBackend.CmdPause );
                break;

            case Job.STEP_OVER:	        
                this.debuggingCommand( job, BTBackend.CmdStepOver );
                break;

            case Job.STEP_INTO:
                this.debuggingCommand( job, BTBackend.CmdStepInto );
                break;

            case Job.STEP_OUT:
                this.debuggingCommand( job, BTBackend.CmdStepOut );
                break;
                
            case Job.RESET:
                this.reset( job );
                break;
                
            case Job.SWITCH_FRAME:
                this.switchFrame( job );
                break;

            case Job.GET_FRAME:
                this.getFrame( job );
            break;
                
            case Job.EVAL:
                this.eval( job );
                break;
                
            case Job.GET_SCRIPTS:
                this.getScripts( job );
                break;
                
            case Job.GET_SOURCE:
                this.getScriptSource( job );
                break;
                
            case Job.PUT_SOURCE:
                this.putScriptSource( job );
                break;
                
            case Job.GET_VARIABLES:
                this.getVariables( job );
                break;
                
            case Job.GET_VARIABLE_VALUE:
                this.getVariableValue( job );
                break;
                
            case Job.SET_VARIABLE_VALUE:
                this.setVariableValue( job );
                break;                                        
                
            case Job.GET_PROFILE_DATA:
                this.getProfileData( job );
                break;
                
            case Job.SET_BREAKPOINTS:
                this.updateBreakpoints( job );
                break;
                
            case Job.REMOVE_BREAKPOINTS:
                this.removeBreakpoints( job );
                break;
                
            case Job.GET_BREAKPOINTS:
                this.getBreakpoints( job );
                break;
                
            case Job.CUSTOM:
            {
				try
				{
					job.quitTask();
				}
				catch( exc )
				{}
            }
            break;
            
            default:
            {
                //
                // job not handled, just quit it
                //
				try
				{
					job.quitTask();
				}
				catch( exc )
				{}
            }
        }
    }

    //-----------------------------------------------------------------------------
    // 
    // onTick(...)
    // 
    // Purpose: Pulse
    // 
    //-----------------------------------------------------------------------------

    session.onTick = function()
    {
        if( this.enabled )
        {
            while( this.pendingBTCalls.length > 0 )
            {
                var btData = this.pendingBTCalls.pop();
                this.postProcessBridgeTalk( btData );
            }
            
            while( this.pendingXMLCalls.length > 0 )
            {
                var xml = this.pendingXMLCalls.pop();
                this.processBreakXML( xml );
            }
        }
    }

    //
    // setup features
    //
    session.setupFeatures();    
}
