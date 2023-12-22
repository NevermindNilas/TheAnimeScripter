function getDefaultSassImplementation() {
  let sassImplPkg = "sass";

  try {
    require.resolve("sass");
  } catch (error) {
    try {
      require.resolve("node-sass");
      sassImplPkg = "node-sass";
    } catch (ignoreError) {
      sassImplPkg = "sass";
    }
  }

  // eslint-disable-next-line import/no-dynamic-require, global-require
  return require(sassImplPkg);
}

const {resolve, dirname, join} = require("path");
const {v4} = require('uuid')
const {renderSync} = getDefaultSassImplementation();

const regexps = {
  sassFile: /([._\-A-Za-z0-9]+)\.s[ac]ss/g,
  sassExt: /\.s[ac]ss$/,
  currentFile: /([A-Za-z0-9]+)\.(t|j)s(x)?/g,
  currentFileExt: /\.(t|j)s(x)?/
};

const isImportDefaultSpecifier = (node) =>
  node.type === 'ImportDefaultSpecifier';

function transformScss(babel) {
  const {types: t} = babel;
  return {
    name: "babel-plugin-transform-scss",
    visitor: {
      ImportDeclaration(path, state) {
        /**
         * Check if file contains scss | sass extension
         */
        if (!regexps.sassExt.test(path.node.source.value)) return;
        const sassFileNameMatch = path.node.source.value.match(
          regexps.sassFile
        );

        /**
         * Get import sass file name and current file name
         */
        const sassFileName = sassFileNameMatch[0].replace(regexps.sassExt, "");
        const file = this.filename.match(regexps.currentFile);
        const filename = `${file[0].replace(
          regexps.currentFileExt,
          ""
        )}_${sassFileName}_${v4()}`;

        /**
         *
         * Get full path of scss file and transpile content
         */
        const scssFileDirectory = resolve(dirname(state.file.opts.filename));
        const fullScssFilePath = join(
          scssFileDirectory,
          path.node.source.value
        );
        const projectRoot = process.env.PWD;
        const nodeModulesPath = join(projectRoot, "node_modules");
        const sassDefaults = {
          file: fullScssFilePath,
          sourceMap: false,
          includePaths: [nodeModulesPath, scssFileDirectory, projectRoot, ...(state.opts.include || [])],
        };
        const sassResult = renderSync({...sassDefaults, ...state.opts});
        const transpiledContent = sassResult.css.toString() || "";

        /**
         * Implement transpiledContent. Creates replacement for "import 'style.scss'" to JS function
         * which renders parsed styles to <style> tag
         */

        const defaultImportNode = path.node.specifiers.find(
          isImportDefaultSpecifier
        );
        if (!defaultImportNode) {
          path.replaceWith(
            t.callExpression(
              t.functionExpression(
                t.identifier(""),
                [],
                t.blockStatement(
                  [
                    t.ifStatement(
                      t.binaryExpression(
                        "===",
                        t.unaryExpression(
                          "typeof", 
                          t.identifier("document")
                        ),
                        t.stringLiteral("undefined")
                      ),
                      t.blockStatement([
                        t.returnStatement()
                      ])
                    ),
                    t.variableDeclaration("const", [
                      t.variableDeclarator(
                        t.identifier("styles"),
                        t.stringLiteral(transpiledContent)
                      )
                    ]),
                    t.variableDeclaration("const", [
                      t.variableDeclarator(
                        t.identifier("fileName"),
                        t.stringLiteral(filename)
                      )
                    ]),
                    t.variableDeclaration("const", [
                      t.variableDeclarator(
                        t.identifier("element"),
                        t.callExpression(
                          t.memberExpression(
                            t.identifier("document"),
                            t.identifier("querySelector")
                          ),
                          [
                            t.stringLiteral(
                              `style[data-sass-component='${filename}']`
                            )
                          ]
                        )
                      )
                    ]),
                    t.ifStatement(
                      t.unaryExpression("!", t.identifier("element"), true),
                      t.blockStatement(
                        [
                          t.variableDeclaration("const", [
                            t.variableDeclarator(
                              t.identifier("styleBlock"),
                              t.callExpression(
                                t.memberExpression(
                                  t.identifier("document"),
                                  t.identifier("createElement")
                                ),
                                [t.stringLiteral("style")]
                              )
                            )
                          ]),
                          t.expressionStatement(
                            t.assignmentExpression(
                              "=",
                              t.memberExpression(
                                t.identifier("styleBlock"),
                                t.identifier("innerHTML")
                              ),
                              t.identifier("styles")
                            )
                          ),
                          t.expressionStatement(
                            t.callExpression(
                              t.memberExpression(
                                t.identifier("styleBlock"),
                                t.identifier("setAttribute")
                              ),
                              [
                                t.stringLiteral("data-sass-component"),
                                t.identifier("fileName")
                              ]
                            )
                          ),
                          t.expressionStatement(
                            t.callExpression(
                              t.memberExpression(
                                t.memberExpression(
                                  t.identifier("document"),
                                  t.identifier("head"),
                                  false
                                ),
                                t.identifier("appendChild"),
                                false
                              ),
                              [t.identifier("styleBlock")]
                            )
                          )
                        ],
                        []
                      ),
                      t.blockStatement(
                        [
                          t.expressionStatement(
                            t.assignmentExpression(
                              "=",
                              t.memberExpression(
                                t.identifier("element"),
                                t.identifier("innerHTML")
                              ),
                              t.identifier("styles")
                            )
                          )
                        ],
                        []
                      )
                    )
                  ],
                  []
                ),
                false,
                false
              ),
              []
            )
          );
        } else {
          const defaultImportName = defaultImportNode.local.name;
          // Replace import with hardcoded value
          path.replaceWith(
            t.variableDeclaration('const', [
              t.variableDeclarator(
                t.identifier(defaultImportName),
                t.stringLiteral(transpiledContent)
              ),
            ])
          );
        }
      }
    }
  };
}

module.exports = transformScss;
