import { readFileSync } from 'fs';
import { extname } from 'path';
import { createFilter } from '@rollup/pluginutils';
import svgToMiniDataURI from 'mini-svg-data-uri';

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

    return isSvg ? svgToMiniDataURI(source) : ("data:" + mime + ";" + format + "," + source);
};

function image(opts) {
  if ( opts === void 0 ) opts = {};

  var options = Object.assign({}, defaults, opts);
  var filter = createFilter(options.include, options.exclude);

  return {
    name: 'image',

    load: function load(id) {
      if (!filter(id)) {
        return null;
      }

      var mime = mimeTypes[extname(id)];
      if (!mime) {
        // not an image
        return null;
      }

      var isSvg = mime === mimeTypes['.svg'];
      var format = isSvg ? 'utf-8' : 'base64';
      var source = readFileSync(id, format).replace(/[\r\n]+/gm, '');
      var dataUri = getDataUri({ format: format, isSvg: isSvg, mime: mime, source: source });
      var code = options.dom ? domTemplate({ dataUri: dataUri }) : constTemplate({ dataUri: dataUri });

      return code.trim();
    }
  };
}

export default image;
//# sourceMappingURL=index.es.js.map
