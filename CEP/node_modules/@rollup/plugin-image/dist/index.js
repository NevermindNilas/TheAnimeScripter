'use strict';

var fs = require('fs');
var path = require('path');
var pluginutils = require('@rollup/pluginutils');
var svgToMiniDataURI = require('mini-svg-data-uri');

function _interopDefaultLegacy (e) { return e && typeof e === 'object' && 'default' in e ? e : { 'default': e }; }

var svgToMiniDataURI__default = /*#__PURE__*/_interopDefaultLegacy(svgToMiniDataURI);

var defaults = {
  dom: false,
  exclude: null,
  include: null
};

var mimeTypes = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp'
};

var domTemplate = function (ref) {
  var dataUri = ref.dataUri;

  return ("\n  var img = new Image();\n  img.src = \"" + dataUri + "\";\n  export default img;\n");
};

var constTemplate = function (ref) {
  var dataUri = ref.dataUri;

  return ("\n  var img = \"" + dataUri + "\";\n  export default img;\n");
};

var getDataUri = function (ref) {
    var format = ref.format;
    var isSvg = ref.isSvg;
    var mime = ref.mime;
    var source = ref.source;

    return isSvg ? svgToMiniDataURI__default['default'](source) : ("data:" + mime + ";" + format + "," + source);
};

function image(opts) {
  if ( opts === void 0 ) opts = {};

  var options = Object.assign({}, defaults, opts);
  var filter = pluginutils.createFilter(options.include, options.exclude);

  return {
    name: 'image',

    load: function load(id) {
      if (!filter(id)) {
        return null;
      }

      var mime = mimeTypes[path.extname(id)];
      if (!mime) {
        // not an image
        return null;
      }

      var isSvg = mime === mimeTypes['.svg'];
      var format = isSvg ? 'utf-8' : 'base64';
      var source = fs.readFileSync(id, format).replace(/[\r\n]+/gm, '');
      var dataUri = getDataUri({ format: format, isSvg: isSvg, mime: mime, source: source });
      var code = options.dom ? domTemplate({ dataUri: dataUri }) : constTemplate({ dataUri: dataUri });

      return code.trim();
    }
  };
}

module.exports = image;
//# sourceMappingURL=index.js.map
