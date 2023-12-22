/**************************************************************************
*
*  @@@BUILDINFO@@@ setupEngine-2.jsx 2.0.1.76  10-Jul-2007
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

// This file contains code that the own engine (the engine which is visible
// to the user) needs to be loaded with.
BridgeTalk.onReceive = function (bt)
{
	return eval (bt.body);
}

function print()
{
	var s = "";
	for (var i = 0; i < arguments.length; i++)
		s += arguments [i];
	_print (s);
}

function log()
{
	if (typeof _log == "function")
	{
		var s = "";
		for (var i = 0; i < arguments.length; i++)
			s += arguments [i];
		_log(s);
	}
}

