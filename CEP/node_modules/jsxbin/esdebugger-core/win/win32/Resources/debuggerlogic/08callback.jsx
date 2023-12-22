/**************************************************************************
*
*  @@@BUILDINFO@@@ 08callback-2.jsx 3.0.4  25-November-2007
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

function Callback( proc, args )
{
    this.proc   = proc;
    this.args   = args;
}

Callback.prototype.call = function( args )
{
    if( this.proc )
    {
        if( typeof this.args == 'undefined' )
            this.proc( args );
        else
            this.proc( this.args, args );
    }
}