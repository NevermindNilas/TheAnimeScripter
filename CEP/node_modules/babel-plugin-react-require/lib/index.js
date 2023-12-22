"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _default;

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance"); }

function _iterableToArrayLimit(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

function _default(_ref) {
  var t = _ref.types;
  var plugin = {
    visitor: {
      Program: {
        enter: function (path, _ref2) {
          var file = _ref2.file;

          // Do nothing if React is already declared
          if (path.scope.hasBinding('React')) {
            return;
          }

          var ourNode = t.importDeclaration([t.importDefaultSpecifier(t.identifier('React'))], t.stringLiteral('react')); // Add an import early, so that other plugins get to see it

          var _path$unshiftContaine = path.unshiftContainer('body', ourNode),
              _path$unshiftContaine2 = _slicedToArray(_path$unshiftContaine, 1),
              newPath = _path$unshiftContaine2[0];

          newPath.get('specifiers').forEach(specifier => {
            path.scope.registerBinding('module', specifier);
          });
          file.set('ourPath', newPath);
        },
        exit: function (_, _ref3) {
          var file = _ref3.file;
          // If our import is still intact and we haven't encountered any JSX in
          // the program, then we just remove it. There's an edge case, where
          // some other plugin could add JSX in its `Program.exit`, so our
          // `JSXOpeningElement` will trigger only after this method, but it's
          // likely that said plugin will also add a React import too.
          var ourPath = file.get('ourPath');

          if (ourPath && !file.get('hasJSX')) {
            if (!ourPath.removed) {
              ourPath.remove();
            }

            file.set('ourPath', undefined);
          }
        }
      },
      ImportDeclaration: function (path, _ref4) {
        var file = _ref4.file;

        // Return early if this has nothing to do with React
        if (path.node.specifiers.every(x => x.local.name !== 'React')) {
          return;
        } // If our import is still intact and we encounter some other import
        // which also imports `React`, then we remove ours.


        var ourPath = file.get('ourPath');

        if (ourPath && path !== ourPath) {
          if (!ourPath.removed) {
            ourPath.remove();
          }

          file.set('ourPath', undefined);
        }
      },
      JSXOpeningElement: function (_, _ref5) {
        var file = _ref5.file;
        file.set('hasJSX', true);
      }
    }
  };

  if (t.jSXOpeningFragment) {
    plugin.visitor.JSXOpeningFragment = (_, _ref6) => {
      var file = _ref6.file;
      file.set('hasJSX', true);
    };
  }

  return plugin;
}